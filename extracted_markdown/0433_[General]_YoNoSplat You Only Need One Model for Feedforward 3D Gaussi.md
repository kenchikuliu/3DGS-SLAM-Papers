# YONOSPLAT: YOU ONLY NEED ONE MODEL FOR FEEDFORWARD 3D GAUSSIAN SPLATTING

Botao Ye1,2 Boqi Chen1,2 Haofei Xu1 Daniel Barath1 Marc Pollefeys1,3 1ETH Zurich 2ETH AI Center 3Microsoft

<!-- image-->  
Figure 1: YoNoSplat, a versatile feedforward model for rapid 3D reconstruction. Given an arbitrary number of unposed and uncalibrated input images covering a wide range of scenes, it predicts 3D Gaussians and can also utilize ground-truth camera poses or intrinsics when available.

## ABSTRACT

Fast and flexible 3D scene reconstruction from unstructured image collections remains a significant challenge. We present YoNoSplat, a feedforward model that reconstructs high-quality 3D Gaussian Splatting representations from an arbitrary number of images. Our model is highly versatile, operating effectively with both posed and unposed, calibrated and uncalibrated inputs. YoNoSplat predicts local Gaussians and camera poses for each view, which are aggregated into a global representation using either predicted or provided poses. To overcome the inherent difficulty of jointly learning 3D Gaussians and camera parameters, we introduce a novel mixing training strategy. This approach mitigates the entanglement between the two tasks by initially using ground-truth poses to aggregate local Gaussians and gradually transitioning to a mix of predicted and ground-truth poses, which prevents both training instability and exposure bias. We further resolve the scale ambiguity problem by a novel pairwise camera-distance normalization scheme and by embedding camera intrinsics into the network. Moreover, YoNoSplat also predicts intrinsic parameters, making it feasible for uncalibrated inputs. YoNoSplat demonstrates exceptional efficiency, reconstructing a scene from 100 views (at 280Ã518 resolution) in just 2.69 seconds on an NVIDIA GH200 GPU. It achieves state-of-the-art performance on standard benchmarks in both pose-free and pose-dependent settings. Our project page is at botaoye.github.io/yonosplat/.

## 1 INTRODUCTION

Feedforward Gaussian Splatting (Charatan et al., 2024; Zhang et al., 2025a) has emerged as a promising direction for accelerating 3D scene reconstruction, directly predicting 3D Gaussian parameters (Kerbl et al., 2023) from input images. This approach bypasses the time-consuming perscene optimization required by methods like NeRF (Mildenhall et al., 2020) and the original 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023). However, the real-world applicability of existing feedforward models is often constrained by restrictive assumptions, such as the need for accurate camera poses (Charatan et al., 2024; Xu et al., 2025), calibrated intrinsics (Ye et al., 2025; Zhang et al., 2025b), or a fixed, limited number of input views (Ye et al., 2025; Chen et al., 2024).

<!-- image-->

Pose-Dependent  
<!-- image-->

Pose-Free  
<!-- image-->  
Pose-Dependent

(b) Teacher-forcing Training  
<!-- image-->  
Pose-Free  
(c) Mix-forcing Training  
Figure 2: Effect of different global Gaussian aggregation strategies during training. (a) Aggregating global Gaussians with predicted camera poses results in poor rendering quality because errors in pose estimation and Gaussian learning compound each other. (b) Using ground-truth poses introduces exposure bias (as indicated by the green arrow: training with ground-truth poses but testing with predicted poses causes misalignment of local Gaussians across different views). (c) Our mix-forcing training achieves high rendering quality in both pose-free and pose-dependent settings.

In practice, scene reconstruction needs to operate under flexible and unconstrained conditions: camera poses may be unavailable or noisy, camera intrinsics unknown, and the number of images may vary significantly. Designing a single model that generalizes across these diverse settings â varying number of views, posed or unposed, calibrated or uncalibrated â remains an open challenge.

In this work, we introduce YoNoSplat, a feedforward model that reconstructs 3D scenes from an arbitrary number of unposed and uncalibrated images, while also seamlessly integrating groundtruth camera information when available. Recent pose-free methods (Ye et al., 2025; Smart et al., 2024) have shown impressive results on sparse inputs (2-4 views) by predicting Gaussians directly into a unified canonical space. However, this approach struggles to scale to a larger number of views. To ensure scalability and versatility, YoNoSplat adopts a different paradigm: it first predicts per-view local Gaussians and their corresponding camera poses, which are then aggregated into a global coordinate system.

This local-to-global design, however, introduces a significant training challenge: the joint learning of camera poses and 3D geometry is highly entangled. Errors in pose estimation can corrupt the learning signal for the Gaussians, and vice-versa. A naive approach that aggregates Gaussians using the modelâs own predicted poses, known as the self-forcing mechanism (Huang et al., 2025), leads to unstable training and poor performance (Fig. 2a). Conversely, the teacher-forcing approach, which relies solely on ground-truth poses for aggregation, decouples the tasks but introduces exposure bias (Ranzato et al., 2015). In this case, the model is never trained on its own imperfect pose predictions, causing performance to degrade at inference when it must depend on them (Fig. 2b). To resolve this dilemma, we propose a novel mix-forcing training strategy. Training begins with pure teacher-forcing to establish a stable geometric foundation. As training progresses, we gradually introduce the modelâs predicted poses into the aggregation step. This curriculum balances stability with robustness, enabling YoNoSplat to operate effectively with either ground-truth or predicted poses at test time (Fig. 2c).

A second fundamental challenge is scale ambiguity, which is particularly pronounced when groundtruth depth is unavailable. This ambiguity arises from two sources: training data poses are often defined only up to an arbitrary scale, and jointly estimating intrinsics and extrinsics is an ill-posed problem without a consistent scale reference. Inspired by NoPoSplat (Ye et al., 2025), which highlighted the importance of camera intrinsics for scale recovery, we develop a pipeline that not only uses intrinsic information but also predicts it, enabling reconstruction from uncalibrated images. To address the data-level ambiguity, we systematically evaluate several scene-normalization strategies and find that normalizing by the maximum pairwise camera distance is most effective, as it aligns with the relative pose supervision used during training (Wang et al., 2025c).

Extensive experiments show that our model, even without ground-truth camera inputs, outperforms prior pose-dependent methods, highlighting the strong geometric and appearance priors learned through our training strategy. The approach generalizes across datasets and varying numbers of views, and reconstructs a complete 3D scene from 100 images in just 2.69 seconds.

Our main contributions are as follows:

â¢ We introduce YoNoSplat, the first feedforward model to achieve state-of-the-art performance in both pose-free and pose-dependent settings for an arbitrary number of views.

â¢ We identify the entanglement of pose and geometry learning as a key challenge and propose a mix-forcing training strategy that effectively mitigates training instability and exposure bias.

â¢ We resolve the scale ambiguity problem through an intrinsic-prediction-and-conditioning pipeline and a pairwise distance normalization scheme, enabling reconstruction from uncalibrated images.

## 2 RELATED WORK

Feedforward 3DGS and NeRF. Original NeRF (Mildenhall et al., 2020), 3DGS (Kerbl et al., 2023), and their variants (Muller et al., 2022; Barron et al., 2021; Ye et al., 2023) require time-consuming Â¨ per-scene optimization. To address this inefficiency, numerous feedforward methods (Yu et al., 2021; Hong et al., 2024; Zhang et al., 2025a; Charatan et al., 2024; Chen et al., 2024) have been proposed. These approaches train neural networks on large-scale datasets to learn geometric and appearance priors, enabling generalization to novel scenes. However, they typically require precise camera poses as input and are restricted to a small number of input views (usually 2â4).

Several works relaxes individual constraints. For instance, Long-LRM (Ziwen et al., 2024) and DepthSplat (Xu et al., 2025) reconstruct scenes from multiple input images through a feedforward network, but still rely on accurate camera poses. Recent pose-free methods (Ye et al., 2025; Zhang et al., 2025b) can reconstruct Gaussian-based scenes from unposed images and even outperform pose-dependent counterparts (Charatan et al., 2024; Chen et al., 2024). Yet, they focus primarily on dual-view inputs; while extendable to more views, they remain limited to scenes with sparse coverage. These methods require known intrinsics and operate with a fixed number of views.

In contrast, our work addresses all these challenges simultaneously: we predict local 3D Gaussians, camera intrinsics, and poses feedforward from an arbitrary number of unposed images. The most similar effort is the concurrent AnySplat (Jiang et al., 2025). However, AnySplat cannot leverage available priors such as intrinsics or extrinsics, whereas our method flexibly incorporates them when present. Furthermore, through a carefully designed training paradigm and pose-normalization strategy, YoNoSplat achieves substantially stronger performance.

Feedforward Point Cloud Prediction. Another line of work closely related to ours is feedforward point cloud prediction models (Wang et al., 2024; 2025a;c). DUSt3R (Wang et al., 2024) demonstrated that a feedforward model trained on large-scale datasets can accurately predict camera intrinsics and scene geometry without requiring optimization. Subsequent works extended this idea to a larger number of input views (Wang et al., 2025a; Yang et al., 2025; Wang et al., 2025c) and to incremental feedforward reconstruction (Wang et al., 2025b). However, these methods cannot be applied to novel view synthesis due to the discontinuous nature of point clouds. Moreover, they all require ground-truth depth supervision during training. In contrast, by replacing point clouds with 3D Gaussians as the scene representation, YoNoSplat enables both novel view synthesis and training on datasets without ground-truth depth (Ling et al., 2024; Zhou et al., 2018).

## 3 METHOD

We introduce YoNoSplat, a method for the feedforward prediction of 3D Gaussians from multiple images. Our approach supports a wide range of scene scales and can optionally utilize available ground truth camera poses and intrinsics.

Problem Formulation. Given V unposed images $( I ^ { v } ) _ { v = 1 } ^ { V }$ as input, where $\pmb { I } ^ { v } \in \mathbb { R } ^ { 3 \times H \times W }$ , our objective is to learn a feedforward network Î¸ that predicts 3D Gaussians representing the underlying scene. By learning geometric and appearance priors from the training data, our method directly reconstructs new scenes without the need for time-consuming optimization. YoNoSplat do this by first predicting per-view local 3D Gaussians that can be transformed into a global scene representation using the given or predicted camera poses $\pmb { p } ^ { v }$ . Specifically, the camera pose parameters are defined as $\pmb { p } ^ { v } = [ \pmb { R } ^ { v } , \pmb { t } ^ { v } ]$ , where $R ^ { v } \in \mathbb { R } ^ { 3 \times 3 }$ denotes the rotation matrix, $\pmb { t } ^ { v } \in \mathbb { R } ^ { 3 }$ represents the translation vector, and [Â·] indicates the concatenation operation. Furthermore, as described in Sec. 3.2, our network also predicts the camera intrinsics $k ^ { v }$ , thus can also eliminate the requirement of camera calibration. Formally, we aim to learn the following mapping:

<!-- image-->  
Figure 3: Overview of YoNoSplat. (a) Features are extracted with a DINOv2 encoder, followed by local-global attention across images, and finally used to predict camera poses and local 3D Gaussians. (b) The Intrinsic Condition Embedding (ICE) module predicts intrinsic parameters $( i . e .$ , focal length), which are then converted into camera rays and re-encoded as conditioning for Gaussian prediction, thereby resolving scale ambiguity.

$$
f _ { \pmb \theta } : \{ ( \pmb I ^ { v } ) \} _ { v = 1 } ^ { V } \mapsto \left\{ \cup \left( \pmb \mu _ { j } ^ { v } , \alpha _ { j } ^ { v } , \pmb r _ { j } ^ { v } , \pmb s _ { j } ^ { v } , \pmb c _ { j } ^ { v } \right) , \pmb k ^ { v } , \pmb p ^ { v } \right\} _ { j = 1 , \dots , H \times W } ^ { v = 1 , \dots , V } .\tag{1}
$$

Here, $( \mu _ { j } , \alpha _ { j } , r _ { j } , s _ { j } , c _ { j } )$ denote Gaussian parameters (Kerbl et al., 2023), representing the center position, opacity, rotation, scale, and color, respectively. All parameters are initially predicted in the local input camera views and can subsequently be transformed into a global representation using either predicted or given camera poses.

## 3.1 ANALYSIS OF THE GAUSSIAN OUTPUT SPACE AND TRAINING STRATEGY

Output Space: Local vs. Canonical Prediction. A fundamental design choice for a feedforward reconstruction model is its output space. Existing methods fall into two main categories. Posefree models such as NoPoSplat (Ye et al., 2025) and Flare Zhang et al. (2025b) predict Gaussians directly in a unified canonical space, which naturally aligns the outputs from all views into a shared coordinate system. In contrast, pose-dependent methods like pixelSplat (Charatan et al., 2024) and MVSplat (Chen et al., 2024) predict Gaussians in a local, per-view space and rely on ground-truth camera poses to transform them into the global world frame.

While canonical-space prediction is effective for a small number of views, its performance degrades as the view count increases (see Tab. 7), an observation consistent with findings in related feedforward point-cloud prediction models (Wang et al., 2025a). To ensure scalability and versatility, YoNoSplat adopts a local prediction paradigm. We architect our model to predict per-view local Gaussians alongside their corresponding camera poses. This design enables our primary goal of pose-free reconstruction by using the predicted poses for aggregation, yet it also retains full compatibility with pose-dependent workflows where ground-truth poses can be supplied. This flexibility is critical for real-world applications, such as map reconstruction, where alignment with a pre-existing, accurate pose distribution is required.

Training Strategy: Mitigating Pose Entanglement. Jointly predicting 3D Gaussians and camera parameters is challenging, as errors in one corrupt the other. Using only predicted poses for aggregation (self-forcing Huang et al. (2025)) tightly couples the tasks, leading to unstable training and degraded performance (Fig. 2, Tab. 5). Using only ground-truth poses (teacher-forcing (Williams & Zipser, 1989)) provides a stable signal but causes exposure bias Ranzato et al. (2015), since the model never trains on its own imperfect predictions. To resolve this dilemma, we introduce a novel mix-forcing training strategy that combines the benefits of both approaches. Our training curriculum begins by exclusively using ground-truth poses (teacher-forcing) to allow the model to learn a stable geometric foundation. After a predefined number of steps, $t _ { \mathrm { s t a r t } } ,$ the probability of using the modelâs predicted poses for aggregation is linearly increased, eventually reaching a final mixing ratio r at step $t _ { \mathrm { e n d } }$ . This strategy effectively mitigates entanglement by first establishing a strong prior for the 3D structure and then gradually adapting the model to both its own predicted distribution and the ground-truth pose distribution, thereby preventing training instability and exposure bias.

## 3.2 MODEL ARCHITECTURE

The overall architecture of YoNoSplat is shown in Fig. 3. We build upon a Vision Transformer (ViT) backbone (Dosovitskiy et al., 2021) and employ a local-global attention mechanism as in

VGGT (Wang et al., 2025a) for robust multi-view feature fusion, which scales more effectively with a large number of input frames than the cross-attention used in prior works (Ye et al., 2025).

Backbone Network. Input images $( I ^ { v } ) _ { v = 1 } ^ { V }$ are divided into patches and flattened into tokens. These image tokens are concatenated with a learnable camera intrinsic token and processed by a ViT encoder with a DINOv2 architecture (Oquab et al., 2023). The encoded features then pass through a decoder consisting of N alternating attention blocks (Wang et al., 2025a;c). Each block contains a per-frame self-attention layer for local feature refinement and a global concatenated self-attention layer where tokens from all views are combined to facilitate cross-frame information flow.

Gaussian Heads. Following (Ye et al., 2025), we use two separate heads to predict the Gaussian centers and all other parameters. Each head consists of M self-attention layers and a final linear layer. To capture fine-grained detail, we upsample the backbone features by a factor of two before feeding them to the heads and add a skip connection from the input image to combat information loss from the ViTâs downsampling.

Pose Head. As discussed in Sec. 3.1, YoNoSplat first predicts local Gaussian parameters and then uses either the given or predicted camera poses to transform them into a unified global coordinate system. The camera head consists of an MLP layer, followed by average pooling and another MLP, to predict a 12D camera vector following (Dong et al., 2025; Wang et al., 2025c). This output vector includes the camera translation $\mathbf { \Delta } \mathbf { \mathcal { t } } ^ { v }$ and a 9D rotation representation (Levinson et al., 2020), which is converted into $\pmb { R } ^ { v }$ using SVD orthogonalization. During training, we follow $\pi ^ { 3 }$ Wang et al. (2025c) and supervise the camera pose with a pairwise relative transformation loss (see Sec. 3.4), ensuring that our model remains invariant to the order of input images.

Intrinsic Head. Predicting camera poses requires cross-frame information and is thus performed during the decoder stage. In contrast, predicting camera intrinsics can be inferred from individual images. Therefore, we perform intrinsic prediction during the encoder stage. Additionally, conditioning on camera intrinsics helps resolve the scale ambiguity problem, as detailed in Sec. 3.3. Specifically, we concatenate an intrinsic token with the input image tokens, which are then processed by the encoder network, allowing the intrinsic token to aggregate image information. This token is subsequently passed through an MLP layer to predict the camera intrinsics.

## 3.3 RESOLVING SCALE AMBIGUITY

Learning to predict Gaussians from video data encounters a scale ambiguity problem, arising from two main factors: (1) training datasets often provide SfM-derived camera poses that are only defined up to an arbitrary scale, and (2) jointly learning camera intrinsics and extrinsics is an ill-posed problem. We address both factors.

Scene Normalization. The ground-truth poses in our training datasets Zhou et al. (2018); Ling et al. (2024) are obtained using SfM methods Schonberger & Frahm (2016), which are only defined up to scale. To avoid scale ambiguity that could hinder learning, it is therefore necessary to normalize the scene during training. Some point-cloud prediction methods Wang et al. (2024; 2025a) normalize scenes using ground-truth depth, but this is not feasible for datasets without depth annotations.

To address this, we propose and evaluate three normalization strategies:

1. Max pairwise distance: given camera centers $\{ c _ { i } \} _ { i = 1 } ^ { N }$ , compute $d _ { i j } \ = \ \| c _ { i } - c _ { j } \| _ { 2 }$ , and set $s = \operatorname* { m a x } _ { i , j } d _ { i j }$ , then normalize $\hat { c } _ { i } = c _ { i } / s$

2. Mean pairwise distance: use $\begin{array} { r } { s = { \frac { 1 } { N ( N - 1 ) } } \sum _ { i \neq j } { \| c _ { i } - c _ { j } \| } . } \end{array}$ 2 and normalize as $\hat { c } _ { i } = c _ { i } / s$

3. Max translation: set $s = \operatorname* { m a x } _ { i } \| c _ { i } \| _ { 2 }$ and normalize $\hat { c } _ { i } = c _ { i } / s$

As shown in Tab. 6, max pairwise distance normalization performs best and is critical to the modelâs success. Since we employ relative camera poses, normalizing by the maximum pairwise distance ensures a consistent scale for camera translations during training.

Intrinsic Condition Embedding (ICE). As demonstrated in (Ye et al., 2025), camera intrinsic information is crucial for resolving scale ambiguity. However, prior work required ground-truth intrinsics at inference time. To remove this dependency, we introduce our Intrinsic Condition Embedding (ICE) module (Fig. 3b). Specifically, intrinsic parameters are first predicted after the encoder stage using the initial intrinsic token, as detailed in Sec.3.2. Subsequently, to implement intrinsic conditioning, the predicted parameters are transformed into camera rays (Ye et al., 2025), passed through a linear layer to obtain embedding features, and then added to the original image features. When ground-truth intrinsics are available, we use them directly to provide more accurate conditioning and thus achieve better performance. In their absence, we instead use the predicted intrinsics for network conditioning. Notably, during training, we condition the network on ground-truth intrinsics rather than the predicted ones. We also experimented with conditioning the decoder on intrinsics predicted by the encoder, but this led to training instability and eventual failure.

<!-- image-->  
Figure 4: Qualitative comparison on DL3DV (Ling et al., 2024). Here we present our results in the pose-free, calibration-free setting, which still produce higher-quality novel view renderings compared to the pose-dependent method DepthSplat (Xu et al., 2025).

## 3.4 MODEL TRAINING

Our models are trained with a multi-task loss as follows:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { i m a g e } } + \lambda _ { \mathrm { i n t r i n } } \mathcal { L } _ { \mathrm { i n t r i n } } + \lambda _ { \mathrm { p o s e } } \mathcal { L } _ { \mathrm { p o s e } } + \lambda _ { \mathrm { o p a c i t y } } \mathcal { L } _ { \mathrm { o p a c i t y } } . } \end{array}\tag{2}
$$

Rendering Loss. Following previous works (Chen et al., 2024; Ye et al., 2025), the rendering loss Limage is set as a linear combination of Mean Squared Error (MSE) and LPIPS (Zhang et al., 2018) loss, which is employed to optimize the Gaussians. During training, we randomly sample 4 target views and render the corresponding images using their ground truth camera poses, and then the rendered images are compared against the ground truth image.

Intrinsic Loss. The intrinsic loss ${ \mathcal { L } } _ { \mathrm { i n t r i n } }$ is used to train the intrinsic prediction head, which is the $l _ { 2 }$ distance between the predicted focal length with the ground truth.

Pose Loss. Following (Wang et al., 2025c), we supervise the pose prediction head with relative pose loss. Specifically, for each pair of input views $( i , j )$ and their predicted pose $( \hat { \bf p } _ { i } , \hat { \bf p } _ { j } )$ , we first calculate their relative pose $\hat { \bf p } _ { i  j } ~ = ~ \hat { \bf p } _ { i } ^ { - 1 } \hat { \bf p } _ { j }$ . The loss is then calculated as $\mathcal { L } _ { \mathrm { p o s e } } ~ =$ $\begin{array} { r } { \frac { 1 } { N ( N - 1 ) } \sum _ { i \neq j } \left( \mathcal { L } _ { \mathrm { R } } ( i , j ) + \lambda _ { \mathrm { t } } \mathcal { L } _ { \mathrm { t } } ( i , j ) \right) } \end{array}$ , where N is the number of input views. Following (Dong et al., 2025; Wang et al., 2025c), the rotation $\mathcal { L } _ { \mathrm { R } }$ and translation losses $\mathcal { L } _ { \mathrm { t } }$ are calculated as:

$$
\mathcal { L } _ { \mathsf { R } } ( i , j ) = \operatorname { a r c c o s } ( ( \operatorname { t r } ( ( \mathbf { R } _ { i  j } ) ^ { \top } \hat { \mathbf { R } } _ { i  j } ) - 1 ) / 2 ) , \mathcal { L } _ { \mathsf { t } } ( i , j ) = \mathcal { H } _ { \delta } ( \hat { \mathbf { t } } _ { i  j } - \mathbf { t } _ { i  j } ) .\tag{3}
$$

Here, tr(Â·) denotes the trace of a matrix, and $\mathcal { H } _ { \delta } ( \cdot )$ calculate the Huber loss.

Opacity Loss. Since a Gaussian is predicted per pixel, the total number of Gaussians grows rapidly as the number of views increases. To mitigate this issue, we apply an opacity regularization loss following (Ziwen et al., 2024) to promote sparsity. Specifically, $\begin{array} { r } { { \mathcal { L } } _ { \mathrm { o p a c i t y } } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \left| o _ { i } \right| } \end{array}$ , where M is the total number of Gaussians. We then prune those with $o _ { i } < \mathrm { { 0 . 0 0 5 } }$ . We observed that this removes around 20%â70% of the Gaussians, depending on the number of images and their overlap.

## 3.5 EVALUATION

For evaluation in the pose-dependent setting, we render the target view using the corresponding ground-truth camera poses. In contrast, under the pose-free setting, the predicted camera space may differ from the ground-truth poses obtained via SfM methods. To faithfully assess the quality of Gaussian reconstruction, we follow prior pose-free approaches (Ye et al., 2025; Wang et al., 2021;

Table 1: Novel view synthesis comparison under various input settings. We report results on DL3DV Ling et al. (2024) with 6, 12, and 24 input views, where p, k, and Opt denote using groundtruth poses, intrinsics, and post-optimization. Our method consistently outperforms previous SOTA approaches, including the pose-dependent DepthSplat, even without prior information.
<table><tr><td rowspan="2">Method</td><td rowspan="2">p</td><td rowspan="2">k</td><td rowspan="2">Opt</td><td colspan="3"> $_ { 6 \mathrm { v } }$ </td><td colspan="3"> $1 2 \mathrm { v }$ </td><td colspan="3">24v</td></tr><tr><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>MVSplat</td><td>â</td><td>V</td><td></td><td>22.659</td><td>0.760</td><td>0.173</td><td>21.289</td><td>0.709</td><td>0.224</td><td>19.975</td><td>0.662</td><td>0.269</td></tr><tr><td>DepthSplat</td><td>â</td><td>â</td><td></td><td>23.418</td><td>0..797</td><td>0.136</td><td>21.911</td><td>0.753</td><td>0.179</td><td>20.088</td><td>0.690</td><td>0.240</td></tr><tr><td> Ours</td><td>â </td><td>â</td><td></td><td>24.717</td><td>0.817</td><td>0.139</td><td>23.285</td><td>0.773</td><td>0.177</td><td>22.664</td><td>0.758</td><td>0.192</td></tr><tr><td>NoPoSplat</td><td></td><td>â</td><td></td><td>22.766</td><td>0.743</td><td>0.179</td><td>19.380</td><td>0.563</td><td>0.318</td><td>17.860</td><td>0.495</td><td>0.397</td></tr><tr><td>Ours</td><td></td><td>â</td><td></td><td>24.887</td><td>0.819</td><td>0.138</td><td>23.149</td><td>0.758</td><td>0.183</td><td>22.354</td><td>0.731</td><td>0.205</td></tr><tr><td>AnySplat</td><td></td><td></td><td></td><td>19.027</td><td>0.554</td><td>0.235</td><td>18.940</td><td>0.549</td><td>0.262</td><td>19.703</td><td>0.596</td><td>0.249</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td>24.531</td><td>0.804</td><td>0.142</td><td>22.933</td><td>0.746</td><td>0.187</td><td>22.174</td><td>0.720</td><td>0.209</td></tr><tr><td>InstantSplat</td><td></td><td></td><td>â</td><td>21.677</td><td>0.627</td><td>0.273</td><td>20.792</td><td>0.580</td><td>0.316</td><td>18.493</td><td>0.510</td><td>0.381</td></tr><tr><td> Ours</td><td></td><td></td><td>â</td><td>27.533</td><td>0.866</td><td>0.106</td><td>26.126</td><td>0.820</td><td>0.133</td><td>25.855</td><td>0.814</td><td>0.136</td></tr></table>

Fan et al., 2024), which first predict the target camera poses and then render the images using these predicted poses for evaluation. The prediction of target camera poses follows (Ye et al., 2025), which optimizes the poses through a photometric loss based on the predicted 3D Gaussians.

Optional Post-Optimization. After YoNoSplat predicts 3D Gaussians and pose parameters, we optionally perform a fast post-optimization. Specifically, we optimize the predicted camera poses along with the Gaussian centers and colors, while keeping all other parameters fixed (see the appendix for details). The results in Tab. 1 show that this optional optimization can further improve performance with a reasonable time cost.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Datasets. We train on RealEstate10K (RE10K) (Zhou et al., 2018) and DL3DV (Liu et al., 2021) using the official splits. RE10K consists of indoor real-estate videos (67,477 train / 7,289 test). DL3DV (Ling et al., 2024) contains 10,000 outdoor videos, 140 for testing. For evaluation on RE10K, we keep test sequences with â¥ 200 frames (1,580 sequences) and use 6 context views due to the smaller scene scale. On DL3DV, we test with (6, 12, 24) input views and maximum frame gaps (50, 100, 150). For generalization, we evaluate the DL3DV-trained model on ScanNet++ (Yeshwanth et al., 2023) by sampling (32, 64, 128) views per sequence with a fixed target view. Inputs are selected by farthest point sampling over camera centers; 8 views are randomly held out as validation.

Implementation Details. YoNoSplat is implemented using PyTorch. The encoder employs the DINOv2 Large model (Oquab et al., 2023) with 24 attention layers, and the decoder consists of 18 alternating-attention layers. The parameters of the backbone, Gaussian center head, and camera pose head are initialized from $\pi ^ { 3 }$ (Wang et al., 2025c), while the remaining layers are initialized randomly. During training, we randomly select the number of input views between 2 and 32 views and sample 4 target views. We train models at two different resolutions, 224 Ã 224 and 280 Ã 518. The 224 Ã 224 model is trained on 16 GH200 GPUs for 150k steps with a batch size of 2 for each, while the 280 Ã 518 model is initialized from the pretrained 224 Ã 224 weights and further trained on 32 GH200 GPUs for another 150k steps with a batch size of 1.

Evaluation Metrics. For the novel view synthesis task, we evaluate with the commonly used metrics: PSNR, SSIM, and LPIPS. For pose estimation, we report the area under the cumulative angular pose error curve (AUC) thresholded at 5â¦, 10â¦, and 20â¦ (Sarlin et al., 2020; Edstedt et al., 2024).

Baselines. We compare against SOTA representative sparse-view generalizable methods on novel view synthesis: 1) Optimization-based: InstantSplat (Fan et al., 2024); 2) Pose-dependent: MVSplat (Chen et al., 2024), DepthSplat (Xu et al., 2025); 3) Pose-free: NoPoSplat (Ye et al., 2025) and AnySplat (Jiang et al., 2025). For relative pose estimation, we compare against SOTA methods: MASt3R (Leroy et al., 2024), VGGT (Wang et al., 2025a), and $\pi ^ { 3 }$ (Wang et al., 2025c).

## 4.2 EXPERIMENTAL RESULTS AND ANALYSIS

Novel View Synthesis. To evaluate our method on complex real-world scenes, we test it on DL3DV with varying numbers of input views, scene scales, and input priors. As shown in Table 1, our model consistently outperforms previous SOTA approaches. Notably, YoNoSplat surpasses leading pose-free methods like NoPoSplat and AnySplat by a substantial margin. More strikingly, even in the most challenging pose-free, intrinsic-free setting, our model outperforms the SOTA posedependent method, DepthSplat, across all view counts. This highlights our modelâs ability to learn powerful priors that compensate for the lack of ground-truth camera information. Qualitatively, Fig. 4 shows that our reconstructions have better cross-view consistency, avoiding the artifacts and inaccurate geometry seen in baselines. The results in Table 1 also reveal that as the number of input views and scene scale increase (e.g., 12 and 24 views), providing ground-truth extrinsics can further improve performance, acknowledging the inherent difficulty of pose estimation in large-scale environments. Furthermore, a fast, optional post-optimization of the predicted Gaussians and poses yields additional performance gains. On the indoor RealEstate10K dataset (Table 2), our 6-view model continues this trend, outperforming both pose-free and pose-dependent SOTA methods, as shown qualitatively in Fig. 5.

<table><tr><td>Method</td><td>b</td><td>k</td><td>PSNR â</td><td>SSIMâ</td><td>LPIPS â</td></tr><tr><td>DepthSplat</td><td>â</td><td>â</td><td>24.156</td><td>0.846</td><td>0.145</td></tr><tr><td>NoPoSplat</td><td></td><td>â</td><td>22.175</td><td>0.750</td><td>0.207</td></tr><tr><td>Ours</td><td>â</td><td>â</td><td>25.037</td><td>0.848</td><td>0.134</td></tr><tr><td>Ours</td><td></td><td>â</td><td>25.395</td><td>0.857</td><td>0.131</td></tr><tr><td>Ours</td><td></td><td></td><td>24.571</td><td>0.823</td><td>0.144</td></tr></table>

<!-- image-->  
GT

<!-- image-->  
Ours

<!-- image-->  
NoPoSplat

<!-- image-->  
DepthSplat  
Table 2: NVS comparison on the RE10k dataset (6 input views) under different prior settings. Our model consistently achieves the best performance.

Figure 5: Qualitative comparison on RealEstate10K Zhou et al. (2018). Our pose-free, calibration-free method enables a more coherent fusion of multi-view contents.  
Ground Truth  
<!-- image-->

32 Views  
<!-- image-->

64 Views  
<!-- image-->  
AnySplat (Trained on ScanNet++)

128 Views  
<!-- image-->

32 Views  
<!-- image-->

64 Views  
<!-- image-->

128 Views  
<!-- image-->  
Figure 6: Qualitative comparison on ScanNet++. YoNoSplat generalizes well to ScanNet++ and demonstrates more coherent fusion of Gaussians across different views compared to AnySplat. Moreover, adding more inputs leads to better rendering quality, as more information is provided.

We recommend that readers watch our supplementary videos for more results.

Cross-Dataset Generalization. To assess generalizability, we train YoNoSplat on the DL3DV dataset and evaluate it on ScanNet++ without fine-tuning. We compare against AnySplat, which is trained on ScanNet++. As shown in Tab, 3, our model significantly outperforms this baseline across all metrics and view counts, despite AnySplatâs training-domain advantage. It is worth noting that our performance consistently improves as more input views are provided, demonstrating our modelâs ability to effectively integrate additional information. The qualitative results in Fig. 6 corroborate this, showing YoNoSplat produces significantly sharper and more coherent reconstructions, demonstrating a better fusion of information across different views. In contrast, the renderings from AnySplat appear blurrier and contain more noticeable artifacts. These findings highlight our modelâs robust ability to generalize to novel datasets not seen during training.

Camera Pose Estimation. As shown in Tab. 4, our model with small resolution input (224 Ã 224) already achieves the best performance compared to state-of-the-art methods, while our model with large input resolution (518 Ã 280) obtains the best performance compared to other state-of-the-art approaches. Moreover, we evaluate our model trained on DL3DV but tested on RealEstate10K (indicated as DL3DVâRE10K in Tab. 4), ensuring that none of the methods are trained on the RealEstate10K dataset. The results demonstrate that our method generalizes well and outperforms all baselines, highlighting that training with a rendering loss also benefits pose estimation.

Table 3: Generalization to ScanNet++. Trained on DL3DV and tested on ScanNet++, our model outperforms AnySplat, despite AnySplat being trained on ScanNet++. Input images are sampled from full sequences with a fixed target view, and our performance improves with more input views.
<table><tr><td rowspan="2">Method</td><td colspan="3">32v</td><td colspan="3">64v</td><td colspan="3">128v</td></tr><tr><td>PSNR â</td><td>SSIMâ</td><td>LPIPS â</td><td>PSNR â</td><td>SSIMâ</td><td>LPIPS â</td><td>PSNR â</td><td>SSIMâ</td><td>LPIPS â</td></tr><tr><td>AnySplat</td><td>14.054</td><td>0.494</td><td>0.468</td><td>15.982</td><td>0.551</td><td>0.412</td><td>16.988</td><td>0.583</td><td>0.386</td></tr><tr><td>Ours w/o GT k</td><td>16.886</td><td>0.600</td><td>0.432</td><td>17.368</td><td>0.608</td><td>0.413</td><td>17.641</td><td>0.617</td><td>0.405</td></tr><tr><td>Ours w/GT k</td><td>17.935</td><td>0.659</td><td>0.380</td><td>18.833</td><td>0.688</td><td>0.342</td><td>19.284</td><td>0.701</td><td>0.325</td></tr></table>

Table 4: Pose estimation comparison. Our method achieves the best pose estimation with a smaller input resolution (224Ã224) and further improves with a larger resolution (518Ã280). We also report zero-shot results on RE10k using a model trained exclusively on DL3DV (so that none of the models are trained on RE10k); our method still outperforms all others.
<table><tr><td rowspan="2">Method</td><td colspan="3">DL3DV</td><td colspan="3">RealEstate10K</td></tr><tr><td>5oâ$</td><td>10Â°â</td><td>20Â° â</td><td>5oâ$</td><td>10Â°â</td><td>20Â°%â</td></tr><tr><td>MASt3R518Ã288</td><td>0.778</td><td>0.883</td><td>0.941</td><td>0.609</td><td>0.776</td><td>0.878</td></tr><tr><td>NoPoSplat256Ã256</td><td>0.538</td><td>0.735</td><td>0.853</td><td>0.443</td><td>0.627</td><td>0.755</td></tr><tr><td>VGGT 518Ã280</td><td>0.700</td><td>0.848</td><td>0.924</td><td>0.566</td><td>0.753</td><td>0.867</td></tr><tr><td>Ï3 518Ã280</td><td>0.795</td><td>0.897</td><td>0.949</td><td>0.705</td><td>0.841</td><td>0.916</td></tr><tr><td>OurS224Ã224</td><td>0.833</td><td>0.917</td><td>0.958</td><td>0.722</td><td>0.852</td><td>0.923</td></tr><tr><td>OurS224Ã224 (DL3DVâRE10K)</td><td></td><td>-</td><td></td><td>0.74</td><td>0.859</td><td>0.924</td></tr><tr><td>OurS518Ã280</td><td>0.844</td><td>0.922</td><td>0.961</td><td>0.813</td><td>0.904</td><td>0.951</td></tr><tr><td>Ours518Ã280 (DL3DVâRE10K)</td><td></td><td>-</td><td></td><td>0.78</td><td>0.884</td><td>0.939</td></tr></table>

Table 5: Mix-forcing achieves the best balance of pose-free and pose-dependent performance.
<table><tr><td rowspan="2">Method</td><td colspan="3">Pose-dependent</td><td colspan="3">Pose-free</td></tr><tr><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td></tr><tr><td>Mix-forcing</td><td>25.212</td><td>0.848</td><td>0.133</td><td>25.587</td><td>0.854</td><td>0.130</td></tr><tr><td>Self-forcing</td><td>24.150</td><td>0.815</td><td>0.150</td><td>24.652</td><td>0.831</td><td>0.145</td></tr><tr><td>Teacher-forcing</td><td>25.228</td><td>0.850</td><td>0.131</td><td>25.300</td><td>0.851</td><td>0.131</td></tr></table>

Table 6: Pose normalization. Max pairwise distance normalization leads to best performance.
<table><tr><td>Norm.</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>maxi,j dij</td><td>25.212</td><td>0.848</td><td>0.133</td></tr><tr><td>meani,j dij</td><td>24.950</td><td>0.845</td><td>0.135</td></tr><tr><td>maxi di</td><td>22.739</td><td>0.756</td><td>0.184</td></tr><tr><td>No Norm.</td><td>22.662</td><td>0.757</td><td>0.185</td></tr></table>

## 4.3 ABLATION STUDIES

For this ablation, we train the model with only 6 input views for faster training, without compromising generalizability. As a result, the performance is slightly better compared with Tab. 5.

Effectiveness of the Mix-Forcing Strategy. We compare mix-forcing with pure teacher-forcing (ground-truth poses) and self-forcing (predicted poses). As shown in Table 5, self-forcing performs worst in both settings, confirming that entangled poseâgeometry learning causes instability. Teacherforcing excels with ground-truth poses but drops under pose-free evaluation due to exposure bias. Mix-forcing balances these trade-offs, achieving the best pose-free results while remaining competitive in the pose-dependent case, yielding a more robust and versatile model.

Importance of Scene Normalization. As discussed in Sec. 3.3, scene normalization is essential for training on datasets with poses that are only defined up-to-scale. Tab. 6 demonstrates this empirically. Without any normalization, the modelâs performance is severely degraded. We compare our chosen strategy, normalizing by the maximum pairwise distance between camera centers, against two alternatives: normalizing by the mean pairwise distance and by the maximum camera translation from the origin. The results clearly indicate that max pairwise distance normalization yields the best performance. This is because it provides a consistent and robust scale reference for camera translations that aligns directly with the relative pose supervision loss used during training.

## 5 CONCLUSION

In this work, we introduced YoNoSplat, a versatile feedforward model for high-quality 3D Gaussian reconstruction from an arbitrary number of images, uniquely capable of operating in both posefree/pose-dependent and calibrated/uncalibrated settings. We address two key challenges: the entanglement of geometry and pose learning, and scale ambiguity. Our novel mix-forcing training strategy resolves the former by balancing training stability and mitigating exposure bias. For the latter, we combine a robust max pairwise distance normalization with an Intrinsic Condition Embedding (ICE) module that enables reconstruction from uncalibrated inputs. These contributions significantly advance the flexibility and robustness of feedforward 3D reconstruction.

## ACKNOWLEDGMENTS

This work was supported as part of the Swiss AI Initiative by a grant from the Swiss National Supercomputing Centre (CSCS) under project ID a144 on Alps. Botao Ye and Boqi Chen are partially supported by the ETH AI Center.

## REFERENCES

Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 5855â5864, 2021.

David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In CVPR, 2024.

Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. arXiv preprint arXiv:2403.14627, 2024.

Siyan Dong, Shuzhe Wang, Shaohui Liu, Lulu Cai, Qingnan Fan, Juho Kannala, and Yanchao Yang. Reloc3r: Large-scale training of relative camera pose regression for generalizable, fast, and accurate visual localization. In CVPR, 2025.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.

Johan Edstedt, Qiyu Sun, Georg Bokman, M Â¨ arten Wadenb Ë ack, and Michael Felsberg. Roma: Robust Â¨ dense feature matching. In CVPR, 2024.

Zhiwen Fan, Wenyan Cong, Kairun Wen, Kevin Wang, Jian Zhang, Xinghao Ding, Danfei Xu, Boris Ivanovic, Marco Pavone, Georgios Pavlakos, et al. Instantsplat: Unbounded sparse-view pose-free gaussian splatting in 40 seconds. arXiv preprint arXiv:2403.20309, 2024.

Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. In ICLR, 2024.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.

Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui Ren, Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang, Feng Zhao, et al. Anysplat: Feed-forward 3d gaussian splatting from unconstrained views. arXiv preprint arXiv:2505.23716, 2025.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George Drettakis. 3d gaussian splat- Â¨ ting for real-time radiance field rendering. ACM TOG, 2023.

Vincent Leroy, Yohann Cabon, and JerÂ´ ome Revaud. Grounding image matching in 3d with mast3r. Ë arXiv preprint arXiv:2406.09756, 2024.

Jake Levinson, Carlos Esteves, Kefan Chen, Noah Snavely, Angjoo Kanazawa, Afshin Rostamizadeh, and Ameesh Makadia. An analysis of svd for deep rotation estimation. In NeurIPS, 2020.

Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In CVPR, 2024.

Andrew Liu, Richard Tucker, Varun Jampani, Ameesh Makadia, Noah Snavely, and Angjoo Kanazawa. Infinite nature: Perpetual view generation of natural scenes from a single image. In ICCV, 2021.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2018.

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020.

Thomas Muller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics prim- Â¨ itives with a multiresolution hash encoding. ACM TOG, 2022.

Maxime Oquab, Timothee Darcet, Th Â´ eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Â´ Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

MarcâAurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. Sequence level training with recurrent neural networks. arXiv preprint arXiv:1511.06732, 2015.

Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue: Learning feature matching with graph neural networks. In CVPR, 2020.

Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In CVPR, 2016.

Brandon Smart, Chuanxia Zheng, Iro Laina, and Victor Adrian Prisacariu. Splatt3r: Zero-shot gaussian splatting from uncalibrated image pairs. arXiv preprint arXiv:2408.13912, 2024.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In CVPR, 2025a.

Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 10510â10522, 2025b.

Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In CVPR, 2024.

Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. Ï3: Scalable permutation-equivariant visual geometry learning. arXiv preprint arXiv:2507.13347, 2025c.

Zirui Wang, Shangzhe Wu, Weidi Xie, Min Chen, and Victor Adrian Prisacariu. Nerfâ: Neural radiance fields without known camera parameters. arXiv preprint arXiv:2102.07064, 2021.

Ronald J Williams and David Zipser. A learning algorithm for continually running fully recurrent neural networks. Neural computation, 1(2):270â280, 1989.

Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. Depthsplat: Connecting gaussian splatting and depth. In CVPR, 2025.

Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 21924â21935, 2025.

Botao Ye, Sifei Liu, Xueting Li, and Ming-Hsuan Yang. Self-supervised super-plane for neural 3d reconstruction. In CVPR, 2023.

Botao Ye, Sifei Liu, Haofei Xu, Xueting Li, Marc Pollefeys, Ming-Hsuan Yang, and Songyou Peng. No pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images. In ICLR, 2025.

Chandan Yeshwanth, Yueh-Cheng Liu, Matthias NieÃner, and Angela Dai. Scannet++: A highfidelity dataset of 3d indoor scenes. In ICCV, 2023.

Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images. In CVPR, 2021.

Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large reconstruction model for 3d gaussian splatting. In ECCV, 2025a.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.

Shangzhan Zhang, Jianyuan Wang, Yinghao Xu, Nan Xue, Christian Rupprecht, Xiaowei Zhou, Yujun Shen, and Gordon Wetzstein. Flare: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views. In CVPR, 2025b.

Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: learning view synthesis using multiplane images. ACM TOG, 2018.

Chen Ziwen, Hao Tan, Kai Zhang, Sai Bi, Fujun Luan, Yicong Hong, Li Fuxin, and Zexiang Xu. Long-lrm: Long-sequence large reconstruction model for wide-coverage gaussian splats. arXiv preprint 2410.12781, 2024.

## A MORE IMPLEMENTATION DETAILS

Training. During the training process, we first randomly sample a clip from the training videos, the context image are sampled with farthest point sampling based on camera centers from training video clips to ensure sufficent input coverage. The target images are randomly sampled from the whole video clip. We employ the AdamW optimizer (Loshchilov & Hutter, 2018), setting the initial learning rate for the backbone to $2 \times 1 0 ^ { - 5 }$ and other parameters to $2 \times 1 0 ^ { - 4 }$ . The weight of intrinsic loss $\lambda _ { \mathrm { i n t r i n } } .$ , pose loss $\lambda _ { \mathrm { p o s e } }$ , and opacity loss are set to 0.5, 0.1, and 0.01 respectively. For the mixforcing training, we set $t _ { s t a r t } = 8 0 k , t _ { e n d } = 1 0 0 k$ , and mixing ratio $r = 0 . 1$

Evaluation. For comparison with baseline methods on novel view synthesis, we use the $2 2 4 \times 2 2 4$ version of our model to ensure a fair comparison, as it best aligns with the experimental settings of the other baselines. Different prior methods adopt different input resolutions (e.g., MVSplat (Chen et al., 2024) and NoPoSplat Chen et al. (2024) use $2 5 6 \times 2 5 6$ , while DepthSplat (Xu et al., 2025) uses $2 5 6 \times 4 4 8 )$ . Due to computational constraints and to avoid noise from in-house reproduction, it is not feasible to retrain all baselines and our model at a unified resolution. However, we have taken care to ensure that the comparisons remain fair and meaningful: 1) Our model has the smallest receptive size among all compared methods. Since all methods first center-crop and then resize, square crops result in minimal receptive coverage. We use this model for novel view synthesis comparisons to maintain fairness. 2) Because our model has the smallest receptive size, we can center-crop and resize the rendered outputs of other methods, ensuring that all comparisons are performed on the same image content.

Optional Post-Optimization. This fast optimization step refines the predicted camera poses, Gaussian centers, and colors for 200 iterations. We use learning rates of 0.005 for pose parameters, 0.0016 for Gaussian means, and 0.0025 for colors. The total optimization time varies with the number of input views: 17.7s for 6 views, 51.1s for 12 views, and 165s for 24 views.

## B MORE EXPERIMENTAL ANALYSIS

## B.1 ON THE UTILITY OF GROUND-TRUTH POSE PRIORS

A noteworthy and somewhat counter-intuitive result emerges from our experiments, as shown in Tables 1 and 2. In settings with a small number of input views (e.g., 6 views), our model operating in the fully pose-free setting outperforms its pose-dependent counterpart, which is supplied with ground-truth camera poses. We hypothesize that this is due to the inherent noise and potential inconsistencies within the âground-truthâ poses themselves, which are typically derived from Structurefrom-Motion (SfM) pipelines. For sparse-view reconstructions, minor inaccuracies in SfM poses can lead to subtle misalignments when aggregating local Gaussians. In contrast, our pose-free model is optimized end-to-end to produce a set of camera poses and a 3D representation that are maximally photometrically consistent with each other. This internal self-consistency can lead to higher-quality renderings than forcing the model to align with a slightly imperfect ground-truth coordinate system. Moreover, the slight misalignment of the target pose also contributes to this.

However, this trend reverses as the number of input views and the scene scale increase (see Tab. 1). For larger view counts (e.g., 12 and 24), the pose-dependent setting regains its advantage. This occurs because pose estimation becomes more challenging as the scene scale increases, whereas SfM-based ground-truth poses provide a strong geometric prior. This analysis highlights the robustness of our model: it can learn priors strong enough to compensate for noisy ground-truth data in sparse-view scenarios, while also effectively leveraging ground-truth pose information when the scene scale is large.

## B.2 ABLATION ON OUTPUT GAUSSIAN SPACE

As discussed in Sec. 3.1, a fundamental design choice is the output representation space. We compare our approach of predicting Gaussians in a local, per-view space against the alternative of predicting them directly into a unified canonical space. Tab. 7 shows that the local prediction strategy significantly outperforms the canonical one on all metrics. This result empirically validates our hypothesis that predicting in a local space is more scalable and robust, especially as the number of views increases, as it avoids the difficulty of forcing a single network to align features from multiple views into one arbitrary coordinate frame.

Table 7: Comparison of Gaussian representations. Local Gaussian performs better on the 6-view setting.
<table><tr><td>Representation</td><td>PSNR â</td><td>SSIMâ</td><td>LPIPS â</td></tr><tr><td>Local Gaussian</td><td>25.587</td><td>0.854</td><td>0.130</td></tr><tr><td>Canonical Gaussian</td><td>24.104</td><td>0.819</td><td>0.172</td></tr></table>

## B.3 IMPACT OF INTRINSIC CONDITION EMBEDDING (ICE)

Table 8: Effect of ICE Module. Using the intrinsics predicted by our model leads to better performance compared to training without intrinsic conditioning.
<table><tr><td></td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>GT Intrinsic</td><td>25.587</td><td>0.854</td><td>0.130</td></tr><tr><td>Pred Intrinsic</td><td>24.711</td><td>0.825</td><td>0.141</td></tr><tr><td>No Intrinsic</td><td>24.481</td><td>0.813</td><td>0.149</td></tr></table>

Table 8 evaluates three inference scenarios: (1) ground-truth intrinsics, (2) predicted intrinsics, and (3) no intrinsic conditioning. Removing intrinsics causes a clear performance drop, confirming their importance for resolving scale ambiguity. Using predicted intrinsics significantly outperforms the no-intrinsic baseline and comes close to ground-truth, demonstrating that ICE enables high-quality reconstruction even from uncalibrated inputs.

## B.4 ABLATION ON PLUCKER RAYS Â¨

Table 9: Quantitative results comparing the effect of adding Plucker ray embedding. Â¨ Our method performs well without pose information as input.
<table><tr><td>Method</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td rowspan="2">w/o PlÃ¼cker w/ PlÃ¼cker</td><td>25.212</td><td>0.848</td><td>0.133</td></tr><tr><td>25.202</td><td>0.851</td><td>0.129</td></tr></table>

Some prior pose-dependent methods (Ziwen et al., 2024; Zhang et al., 2025a) incorporate rich geometric ray representations, such as Plucker coordinates, to help the network align multi-view infor- Â¨ mation. We investigate whether such explicit geometric cues are necessary for our model. Tab. 9 shows that incorporating a Plucker ray embedding provides no significant performance benefit; the Â¨ metrics are nearly identical to our default model. This result demonstrates that our model learns to effectively establish cross-view correspondence and align local Gaussians using image features alone, without relying on explicit geometric embeddings. This simplifies the architecture and reinforces its suitability for pose-free scenarios where such information may not be readily available or reliable.

## C LIMITATIONS

Our method leverages a feedforward approach to reconstruct wide-coverage scenes from an arbitrary number of unposed images. However, the maximum number of input views is constrained by GPU memory. Therefore, an interesting future direction is to explore incremental feedforward reconstruction (Wang et al., 2025b). Moreover, as shown in Tab. 1, pose optimization can still substantially improve the performance of our Gaussians, indicating that the current feedforward model has significant potential for further enhancement.

32 Views  
64 Views  
128 Views  
32 Views  
64 Views  
128 Views  
<!-- image-->  
Figure 7: More qualitative comparison on ScanNet++. YoNoSplat demonstrates strong generalization to the unseen ScanNet++ dataset, producing more coherent reconstructions than AnySplat by better fusing multi-view Gaussians. The quality improves as more input views are provided (left to right). Notably, while our model performs well without priors (Ours w/o p.), providing ground-truth intrinsics (Ours w/ p.) further enhances generalization and fusion, leading to the highest fidelity results.

## D MORE VISUAL COMPARISONS

Here, we provide more qualitative comparisons on the ScanNet++ (Yeshwanth et al., 2023), RealEstate10K (Zhou et al., 2018), and DL3DV (Ling et al., 2024) datasets. As shown in Fig. 7, Fig. 8, and Fig. 9, our pose-free method consistently outperforms the previous SOTA pose-free method, NoPoSplat (Ye et al., 2025). Moreover, we can even achieve superior novel view rendering quality compared to SOTA pose-required methods (Chen et al., 2024; Xu et al., 2025) and optimization-based methods (Fan et al., 2024).

## E USE OF LARGE LANGUAGE MODELS

We use LLMs solely for improving grammar, wording, and overall readability of the manuscript. The model is not used for ideation, experimental design, implementation, or analysis. All technical content, methodology, and results are original and developed entirely by the authors.

<!-- image-->  
Input Views  
Ours  
NoPoSplat  
DepthSplat

Figure 8: Qualitative comparison on RealEstate10K (Zhou et al., 2018). Our method achieves high-quality novel view synthesis compared with the previous SOTA pose-required method (Xu et al., 2025) and pose-free method (Ye et al., 2025).

<!-- image-->  
Figure 9: Qualitative comparison on DL3DV (Ling et al., 2024). We compare our method with state-of-the-art optimization-based (Fan et al., 2024), pose-required (Chen et al., 2024; Xu et al., 2025), and pose-free (Ye et al., 2025; Jiang et al., 2025) methods. Here, the results of our method are obtained under the pose-free, intrinsic-free setting. The results demonstrate that our method generates novel view images of higher quality than these methods.