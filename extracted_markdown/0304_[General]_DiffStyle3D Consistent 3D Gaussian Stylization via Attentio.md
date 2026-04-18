# DiffStyle3D: Consistent 3D Gaussian Stylization via Attention Optimization

Yitong Yang1, Xuexin Liu1, Yinglin Wang1\*, Jing Wang1, Hao Dou1, Changshuo Wang2, Shuting He1\*

1School of Computing and Artificial Intelligence, Shanghai University of Finance and Economics 2Department of Computer Science, University College London, University of London yangyitong@stu.sufe.edu.cn, {wang.yinglin, shuting.he}@sufe.edu.cn

<!-- image-->  
Figure 1: Our method enables high-quality 3D stylization across diverse styles for both scenes and objects.

## Abstract

3D style transfer enables the creation of visually expressive 3D content, enriching the visual appearance of 3D scenes and objects. However, existing VGG- and CLIP-based methods struggle to model multi-view consistency within the model itself, while diffusion-based approaches can capture such consistency but rely on denoising directions, leading to unstable training. To address these limitations, we propose Diff-Style3D, a novel diffusion-based paradigm for 3DGS style transfer that directly optimizes in the latent space. Specifically, we introduce an Attention-Aware Loss that performs style transfer by aligning style features in the self-attention space, while preserving original content through content feature alignment. Inspired by the geometric invariance of 3D stylization, we propose a Geometry-Guided Multi-View Consistency method that integrates geometric information into self-attention to enable cross-view correspondence modeling. Based on geometric information, we additionally construct a geometry-aware mask to prevent redundant optimization in overlapping regions across views, which further improves multi-view consistency. Extensive experiments

show that DiffStyle3D outperforms state-of-the-art methods, achieving higher stylization quality and visual realism.

## 1 Introduction

92; 193 With the rapid development of applications such as virtual reality, gaming, and film production, the demand for digital content is shifting from 2D images to 3D representations, making large-scale, high-quality 3D assets increasingly essential (He et al. 2025). Against this backdrop, 3D stylization has emerged as a promising research direction, aiming to transform static 3D geometric representations into expressive digital assets with distinctive aesthetic characteristics, thereby facilitating the low-cost, efficient, and scalable creation of high-quality 3D artistic content. Previous 3D style transfer methods (Chen et al. 2024; Liu et al. 2023a; Fujiwara, Mukuta, and Harada 2024; Zhang et al. 2022) predominantly relied on NeRF-based (Mildenhall et al. 2021) representations, which suffer from substantial computational overhead and long training times, limiting their efficiency and scalability. Recently, 3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) has emerged as a promising alternative, offering significantly improved rendering efficiency and high visual quality. As a result, 3DGS has quickly become a focal point in 3D style transfer research.

Currently, 3DGS-based style transfer research (Zhuang et al. 2025; Gu et al. 2024; Lin, Lei, and Jia 2025; Zhang et al. 2024) can be broadly divided into three categories. First, VGG-based methods (Liu et al. 2024; Galerne et al. 2025; Lin, Lei, and Jia 2025; Saroha et al. 2024), inspired by 2D feature statistic matching (Gatys, Ecker, and Bethge 2016; Jing et al. 2019), enforce style consistency by minimizing Gram matrix discrepancies. While these methods offer stable training, inherent model limitations hinder multi-view consistency, often resulting in inconsistent stylization across varying viewpoints. Second, CLIPbased methods (Howil et al. 2025; Kovacs, Hermosilla, and Â´ Raidou 2024) align feature directions within the CLIP embedding space to introduce semantic style constraints; however, they similarly fail to explicitly model cross-view correspondences, leading to stylistic drift or flickering. Recently, diffusion-based methods (Yang et al. 2026) leverage the intrinsic properties of diffusion models to establish multi-view consistency. Nevertheless, these approaches frequently suffer from unstable training and potential artifacts, such as over-smoothing, due to their reliance on optimization along predicted denoising directions.

To address these challenges, we propose DiffStyle3D, a novel diffusion-based paradigm for 3DGS stylization. Unlike previous approaches that rely on denoising directions from diffusion models for optimization, we introduce an Attention-Aware Loss that achieves stable 3D style transfer through direct latent-space optimization while effectively preserving the original content. It consists of two terms. (1) Style loss: within self-attention, we inject the keys (K) and values (V) from the style image into the original queries (Q), using the resulting attention output as a stylization signal to guide the integration of style information into the 3D representation. (2) Content loss: to preserve content fidelity, we align the attention outputs of the content image with those of the rendered image.

Motivated by the geometric invariance of 3D stylization, where only color-related parameters are optimized while geometry remains fixed, we propose Geometry-Guided Multi-View Consistency. By leveraging camera parameters and depth maps, we explicitly determine geometric relationships and incorporate this information into the self-attention mechanism to form Geometry-Guided Attention, thereby modeling cross-view correspondences and mitigating view conflicts caused by inconsistent style information. Additionally, based on the geometric information, we introduce a geometry-aware mask to prevent redundant optimization in multi-view overlapping regions, further improving multiview consistency. In summary, our key contributions are as follows:

â¢ To the best of our knowledge, DiffStyle3D is the first paradigm to perform 3DGS stylization by optimizing directly in the latent space of a diffusion model.

â¢ We propose an Attention-Aware Loss that enables style transfer while preserving the original content.

â¢ We propose Geometry-Guided Multi-View Consistency to mitigate multi-view inconsistency.

â¢ Extensive experiments demonstrate that our method outperforms existing state-of-the-art approaches in both qualitative and quantitative evaluations.

## 2 Related Work

2D Style Transfer. Style transfer has remained a central topic in generative visual research, aiming to map the stylistic characteristics of a reference image onto a content image. The early pioneering approach (Gatys, Ecker, and Bethge 2016) achieved neural style transfer by minimizing the distance between Gram matrices derived from VGG features, which motivated extensive follow-up works (Heitz et al. 2021; Risser, Wilmot, and Barnes 2017; Vacher et al. 2020). With the rapid development of diffusion models, existing methods have increasingly relied on this framework, leading to the emergence of numerous fine-tuning-based approaches (Zhou et al. 2025; Ye et al. 2023; Xing et al. 2024; Yang et al. 2025a) and training-free methods (He et al. 2024; Xu et al. 2024; Wang et al. 2024). Compared to traditional VGG-feature-based methods, diffusion-driven approaches achieve higher content fidelity and improved stylistic expressiveness. Therefore, we propose DiffStyle3D, a fully diffusion-based framework.

Attention Control. As one of the most advanced paradigms in generative modeling, diffusion models (Ho, Jain, and Abbeel 2020; Nichol and Dhariwal 2021; Rombach et al. 2022) have demonstrated remarkable capabilities across both 2D and 3D domains (Podell et al. 2023; Liu et al. 2023b; Shi et al. 2023). At the heart of these models lies the attention mechanism, which has been extensively explored in recent research (Alaluf et al. 2024; Cao et al. 2023). By imposing various forms of control on self-attention and cross-attention modules, existing methods have achieved superior performance in tasks such as content editing (Yang et al. 2025b; Hertz et al. 2022) and style transfer (Hertz et al. 2024; Chung, Hyun, and Heo 2024). Based on the selfattention mechanism, we propose an Attention-Aware Loss to effectively transfer stylistic information while preserving the original content.

3DGS Style Transfer. Following a trajectory similar to that of 2D style transfer, style transfer methods for 3D Gaussian Splatting can be broadly categorized into three groups. Most existing approaches are VGG-based methods, which optimize the 3D scene using feature matching losses (Liu et al. 2024; Saroha et al. 2024), multi-scale losses (Galerne et al. 2025), or nearest-neighbor feature matching losses (Jain et al. 2024; Zhang et al. 2024). Another line of CLIP-based methods (Howil et al. 2025; Kovacs, Hermosilla, and Raidou Â´ 2024) achieves stylization by aligning representations in the CLIP embedding space. More recently, diffusion-based methods (Zhuang et al. 2025) have been explored for 3D stylization, either by generating stylized 2D image supervision using diffusion models (Gu et al. 2024; Yu et al. 2024), which essentially reformulates the problem as a 2D style transfer task, or by directly distilling diffusion models into the 3D representation (Yang et al. 2026). However, these approaches struggle to effectively establish multi-view consistency. In contrast, we introduce Geometry-Guided Multi-View Consistency to enforce cross-view consistency.

<!-- image-->  
Figure 2: Overview of DiffStyle3D. We introduce an Attention-Aware Loss that enables style transfer while preserving content. To model multi-view correspondences, we derive a explicit geometry guidance from camera parameters and depth maps and incorporate it into Self-Attention (SA) to form Geometry-Guided Attention (GGA). Additionally, a geometry-aware mask $\mathcal { M } _ { G }$ restricts optimization to non-overlapping regions, further improving multi-view consistency.

## 3 Preliminary

Self-Attention. Given an input sequence $\boldsymbol { X } \in \mathbb { R } ^ { n \times d }$ , th e self-attention mechanism projects each token into a triplet of latent representations consisting of query Q, key K, and value V through learned linear transformations. The attention output is computed as

$$
\mathrm { A t t n } ( Q , K , V ) = \mathrm { S o f t m a x } \left( \frac { Q K ^ { \top } } { \sqrt { d _ { k } } } \right) V ,\tag{1}
$$

where $d _ { k }$ is the scaling factor. This mechanism allows each token to aggregate information from others according to pairwise relevance, capturing contextual and contentdependent interactions.

3D Gaussian Splatting. 3DGS represents a 3D scene as a set of M anisotropic Gaussians, formulated as:

$$
\operatorname* { m i n } _ { \Theta } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathcal { L } ( \mathcal { R } ( C _ { i } ; \Theta ) , I _ { i } ^ { g t } ) ,\tag{2}
$$

where $\Theta = \{ ( \mu _ { m } , \Sigma _ { m } , \alpha _ { m } , \mathcal { C } _ { m } ) \} _ { m = 1 } ^ { M }$ denotes the 3D Gaussian parameters. Here, $\mu _ { m } , \Sigma _ { m } , \alpha _ { m } .$ , and $\mathcal { C } _ { m }$ represent the mean position, covariance matrix, opacity, and spherical harmonics (SH) coefficients for color, respectively. $C _ { i }$ denotes the i-th camera parameters, $\mathcal { R } ( \cdot )$ the rasterization-based renderer, and $I _ { i } ^ { g t }$ the corresponding ground-truth image.

## 4 Method

Given a style image Is, we aim to transfer its style to the 3D scene while preserving the original content. Therefore, we only optimize the color-related parameters. We propose DiffStyle3D (Fig. 2), which employs an Attention-Aware Loss for 3D style transfer via direct latent-space optimization (Sec. 4.1). To capture cross-view correspondences and improve multi-view consistency, we introduce Geometry-Guided Multi-View Consistency (Sec. 4.2).

## 4.1 Attention-Aware Loss

Previous diffusion modelâbased approaches for 3D style transfer (Yang et al. 2026; Zhuang et al. 2025) typically optimize 3D representations by following the predicted denoising directions. However, this optimization strategy often leads to instability and overly smooth results, limiting their effectiveness in high-quality 3D stylization. Drawing inspiration from inference-time optimization techniques (Chen, Laina, and Vedaldi 2024; Ding, Mao, and Aizawa 2024; Shi et al. 2024), we introduce an Attention-Aware Loss that establishes a new 3D stylization paradigm through direct optimization in the latent space of diffusion models. By aligning representations in the self-attention feature space, our method enables accurate style transfer while preserving original content, without relying on unstable denoising guidance.

Specifically, given a 3D Gaussian scene and a style image $I ^ { s }$ , we sample N cameras $C _ { N } = \{ c _ { 1 } , c _ { 2 } , . . . , c _ { N } \}$ in each batch to render the scene, producing a set of rendered images $I _ { N }$ along with their corresponding original content images $I _ { N } ^ { c }$ . These images are then fed into a diffusion model to extract features, which can be formulated as follows:

<!-- image-->  
Figure 3: Results with different stylization signals. We conduct experiments using a fixed viewpoint of the 3D scene. Directly using the attention outputs of the style image as stylization signals leads to severe content leakage.

$$
\begin{array} { l l l } { { z _ { N } = E ( I _ { N } ) , } } & { { z _ { N } ^ { c } = E ( I _ { N } ^ { c } ) , } } & { { z ^ { s } = E ( I ^ { s } ) , } } \\ { { h _ { N } : = \epsilon _ { \theta } ( z _ { N } ) , } } & { { h _ { N } ^ { c } : = \epsilon _ { \theta } ( z _ { N } ^ { c } ) , } } & { { h ^ { s } : = \epsilon _ { \theta } ( z ^ { s } ) , } } \end{array}\tag{3}
$$

where $E ( \cdot )$ denotes the VAE encoder, $\epsilon _ { \theta }$ represents the UNet. $h _ { N } , h _ { N } ^ { c }$ , and $h ^ { s }$ denote the features extracted from specific layers of the UNet, which are used to compute the style and content losses.

Style Loss. Controlling self-attention has been widely adopted in style transfer tasks, motivating us to design our loss function around the self-attention mechanism. A straightforward solution is to directly align the attention outputs of the style image and the rendered image to achieve style transfer. However, such a strategy often leads to severe content leakage from the style image, as illustrated in Fig. 3. To address this issue, we inject stylistic semantics by combining the key (K) and value (V) from the style image with the query (Q) from the rendered image and use the resulting attention output as the stylization signal. Formally, we first extract the $Q _ { N } , K _ { N }$ , and $V _ { N }$ from the self-attention layers of the rendered image to compute the attention output, which are centered to zero mean and then normalized:

$$
\widehat { \mathcal { A } } _ { N } = \frac { \mathcal { A } _ { N } - \mu ( \mathcal { A } _ { N } ) } { \vert \vert \mathcal { A } _ { N } - \mu ( \mathcal { A } _ { N } ) \vert \vert _ { 2 } } , \mathcal { A } _ { N } = \mathrm { A t t n } ( Q _ { N } , K _ { N } , V _ { N } ) ,\tag{4}
$$

where $\mu ( \cdot )$ denotes the mean over channels. Meanwhile, $K ^ { s }$ and $V ^ { s }$ from the style image are integrated with $Q _ { N }$ to inject style semantics:

$$
\boldsymbol { \widehat { \mathcal { A } } ^ { s } } = \frac { \boldsymbol { \mathcal { A } ^ { s } } - \mu ( \boldsymbol { \mathcal { A } ^ { s } } ) } { \| \boldsymbol { \mathcal { A } ^ { s } } - \mu ( \boldsymbol { \mathcal { A } ^ { s } } ) \| _ { 2 } } , \boldsymbol { \mathcal { A } ^ { s } } = \mathrm { A t t n } ( \boldsymbol { Q } _ { N } , \boldsymbol { K ^ { s } } , \boldsymbol { V ^ { s } } ) .\tag{5}
$$

Finally, style guidance is achieved by minimizing the distance between the two representations:

$$
\mathcal { L } _ { s } = \left. \widehat { \mathcal { A } } _ { N } - \widehat { \mathcal { A } } ^ { s } \right. _ { 2 } ^ { 2 } .\tag{6}
$$

Directly applying $\ell _ { 1 }$ or $\ell _ { 2 }$ loss on $\mathcal { A } ^ { s }$ and $\mathcal { A } _ { N }$ focuses on the absolute values of the features, which can slow down training. By centering and normalizing features before computing the loss, we emphasize the consistency of their direction and patterns, enabling a more effective style transfer. Content Loss. A central challenge in style transfer is to effectively apply the target style while preserving the original content. To prevent over-stylization that could distort local features or semantic structures in the rendered image, we design a content loss. Similar to the style loss, the content loss is defined based on self-attention. It preserves the original content by minimizing the distance between the attention representations of the content image and the rendered image. Formally, it can be expressed as follows:

<!-- image-->  
Figure 4: Results obtained using different timestep during optimization. Random denotes randomly sampled timestep throughout the optimization process, while decreasing simulates the diffusion process by progressively decreasing the time step from $T$ to 0.

$$
\widehat { \pmb { A } } _ { N } ^ { c } = \frac { \pmb { A } _ { N } ^ { c } - \mu ( \pmb { A } _ { N } ^ { c } ) } { \lVert \pmb { A } _ { N } ^ { c } - \mu ( \pmb { A } _ { N } ^ { c } ) \rVert _ { 2 } } , \pmb { A } _ { N } ^ { c } = \mathrm { A t t n } ( \pmb { Q } _ { N } ^ { c } , \pmb { K } _ { N } ^ { c } , V _ { N } ^ { c } ) ,
$$

$$
\mathcal { L } _ { c } = \left. \widehat { \mathcal { A } } _ { N } - \widehat { \mathcal { A } } _ { N } ^ { c } \right. _ { 2 } ^ { 2 } .\tag{7}
$$

(8)

Timestep Choice. As illustrated in Fig. 4, we analyze the effect of different fixed diffusion timesteps during optimization. Larger timesteps introduce increased noise and result in blurred stylization, while smaller timesteps better preserve fine-grained stylistic details and brushstroke textures. We further consider random and decreasing timestep strategies commonly used in 3D generation. However, since these strategies still involve large timesteps, they tend to introduce blurring artifacts. As a result, we adopt a fixed timestep of t = 1 as a key choice in our method.

## 4.2 Geometry-Guided Multi-View Consistency

Although the Attention-Aware Loss achieves promising results in style transfer, the same object may still receive inconsistent stylistic representations across different views, leading to noticeable cross-view inconsistency. To address this issue, we propose Geometry-Guided Multi-View Consistency. Unlike VGG-based (Galerne et al. 2025) and CLIPbased methods (Howil et al. 2025), which struggle to model such cross-view constraints within the model, our approach leverages the intrinsic self-attention mechanism of diffusion models to capture correlations across different views, thereby improving multi-view consistency.

Explicit Geometry Guidance. In our framework, we optimize only color-related parameters, ensuring geometric invariance of the 3D Gaussians and a fixed depth map $D _ { b }$ for any given viewpoint $b .$ This stability allows us to explicitly establish geometric correspondences across views using known camera intrinsics and poses. For a pixel p in the reference view b, its corresponding sampling coordinate in source view $j ,$ denoted as $\mathbf { g } _ { b  j } ( \mathbf { p } )$ , is derived via back-projection and re-projection:

$$
{ \bf g } _ { b  j } ( { \bf p } ) = \Pi \big ( { \bf K } _ { j } { \bf T } _ { j } ^ { w 2 c } { \bf T } _ { b } ^ { c 2 w } { \cal D } _ { b } ( { \bf p } ) { \bf K } _ { b } ^ { - 1 } \tilde { \bf p } \big ) ,\tag{9}
$$

where $\tilde { \mathbf { p } }$ represents the homogeneous coordinates of $\mathbf { p } ,$ while K and T denote camera intrinsics and extrinsics. w2c and c2w denote the world-to-camera and camera-to-world transformations, respectively. $\Pi ( \cdot )$ represents perspective projection followed by normalization to the [â1, 1] sampling space. To account for occlusions and boundaries, we define a visibility mask:

$$
\mathbf { v } _ { b  j } ( \mathbf { p } ) = \mathbf { 1 } \big ( \mathrm { i n F r o n t } _ { j } ( \mathbf { p } ) \ \land \ \mathbf { g } _ { b  j } ( \mathbf { p } ) \in \Omega \big ) ,\tag{10}
$$

where $\mathbf { 1 } ( \cdot )$ denotes the indicator function, â¦ denotes the valid image domain, and $\mathrm { i n F r o n t } _ { j } ( \cdot )$ enforces that the reprojected point has a positive depth in view j.

Geometry-Guided Attention. We integrate the obtained sampling grids and visibility masks into the self-attention mechanism of the diffusion model to explicitly model correspondences across multiple views, thereby strengthening cross-view consistency constraints. Specifically, we augment K and V of the reference view by warping features from all other views within the batch. For a batch of N views, $K _ { b } ^ { \prime }$ and $V _ { b } ^ { \prime }$ for view b are formulated as:

$$
\begin{array} { r l } & { K _ { b } ^ { \prime } = [ K _ { b } ; \{ \mathcal { W } _ { b  j } ( K _ { j } ) \mid j \in \{ 0 , \ldots , N - 1 \} , \ j \neq b \} ] , } \\ & { V _ { b } ^ { \prime } = [ V _ { b } ; \{ \mathcal { W } _ { b  j } ( V _ { j } ) \mid j \in \{ 0 , \ldots , N - 1 \} , \ j \neq b \} ] , } \end{array}\tag{11)(}
$$

where $[ \cdot ; \cdot ]$ denotes the concatenation and $\mathcal { W } _ { b  j } ( \cdot )$ represents the bilinear warping operator guided by $\mathbf { g } _ { b  j }$ . By rewriting $\mathcal { A } _ { N }$ in Eq. 4, the Geometry-Guided Attention (GGA) formula is defined as:

$$
\mathrm { A t t n } ( Q _ { N } , K _ { N } ^ { \prime } , V _ { N } ^ { \prime } ) = \mathrm { S o f t m a x } \left( \frac { Q _ { N } { K _ { N } ^ { \prime } } ^ { \top } } { \sqrt { d _ { k } } } + { \bf M _ { v } } \right) V _ { N } ^ { \prime } ,\tag{12}
$$

where $K _ { N } ^ { \prime } = \{ K _ { b } ^ { \prime } \} _ { b = 0 } ^ { N - 1 } , V _ { N } ^ { \prime } = \{ V _ { b } ^ { \prime } \} _ { b = 0 } ^ { N - 1 } . \mathbf { M _ { v } }$ consists of visibility masks v and serves as the attention mask to prevent erroneous feature aggregation from occluded regions.

Geometry-Aware Mask. To avoid redundant optimization over multi-view overlapping regions, we introduce a geometry-aware mask $\mathcal { M } _ { G }$ , which is defined as follows:

$$
\mathcal { M } _ { G } : = \{ \mathbf { 1 } ( \forall j < b , \mathbf { v } _ { b  j } ( \mathbf { p } ) = 0 ) \} _ { b , \mathbf { p } } .\tag{13}
$$

$\mathcal { M } _ { G }$ is a collection of masks, one for each view. For a given view $b ,$ its mask retains only the pixels that have not been observed in any previous view $j ~ < ~ b ,$ , setting them to 1. Consequently, $\mathcal { M } _ { G }$ represents the non-overlapping regions across all views in the current batch, which further improves multi-view consistency.

Optimization Objective. We extract features from all selfattention layers of the diffusion model. The final loss is defined as follows:

$$
\mathcal { L } = \mathcal { M } _ { G } \cdot ( \mathcal { L } _ { s } + \lambda \mathcal { L } _ { c } ) .\tag{14}
$$

where Î» is a scaling factor that controls the strength of content preservation during style transfer. Our method is detailed in Alg. 1.

Algorithm 1: DiffStyle3D   
Require: VAE encoder $E ,$ diffusion model $\epsilon _ { \theta } { \mathrm { . } }$ , GGA  
integrated diffusion model $\epsilon _ { \theta } ^ { \prime } ,$ training iterations S,   
fixed timestep $t , N$ views per batch, style image $I ^ { s } .$   
1: for $s = 1$ to S do   
2: Sample: $I _ { N } ~ = ~ \mathcal { R } ( C _ { N } ; \Theta ) , ~ I _ { N } ^ { c } .$ , depth maps $D _ { N }$ ,   
camera $C _ { N }$ intrinsics and extrinsics.   
3: Explicit Geometry Guidance: $\mathbf { g }$ and v defined by   
Eq. 9, Eq. 10, geometry-aware mask $\mathbf { M } _ { G }$   
4: $z _ { N } , z _ { N } ^ { c } , z ^ { s } \gets E ( \bar { I _ { N } } ) , E ( I _ { N } ^ { c } ) , E ( I ^ { s } )$   
$\widehat { A } _ { N } ^ { c } , Q _ { N } \gets h _ { N } ^ { c } : = \epsilon _ { \theta } ( z _ { N } ^ { c } , t ) ,$ in Eq. 7   
5: $\widehat { \cal A } ^ { s }  \{ Q _ { N } , \quad h ^ { s } : = \epsilon _ { \theta } ( z ^ { s } , t ) \} .$ in Eq. 5   
$\widehat { A } _ { N } \gets h _ { N } : = \epsilon _ { \theta } ^ { \prime } ( z _ { N } , t ) ,$ in Eq. 4, 11, 12   
6: $\mathcal { L } = \mathcal { M } _ { G } ( \mathcal { L } _ { s } ( \widehat { \mathcal { A } } _ { N } , \widehat { \mathcal { A } } ^ { s } ) + \lambda \mathcal { L } _ { c } ( \widehat { \mathcal { A } } _ { N } , \widehat { \mathcal { A } } _ { N } ^ { c } ) )$ Eq. 6, 8   
7: Compute $\nabla _ { z _ { N } } \mathcal { L }$   
8: Update 3D Gaussians using $\nabla _ { z _ { N } } \mathcal { L }$   
9: end for   
Ensure: 3D Stylized Scene.

## 5 Experiments

Datasets. We select 8 scenes from the Tandt DB dataset (Kerbl et al. 2023) and the Mip-NeRF 360 dataset (Barron et al. 2022). Each scene is stylized using 14 different style images, resulting in a total of 112 stylization experiments. In addition, we employ SAM3D (Chen et al. 2025) to extract 10 individual objects, producing 140 object-level stylization results in total. We comprehensively evaluate our method at both the scene-level and the objectlevel across diverse artistic domains.

Metrics. We evaluate content preservation using CLIP-C (Radford et al. 2021) and FID, while CLIP-S is used to assess style transfer quality. CLIP-CONS and CLIP-F (Howil et al. 2025) are employed to measure semantic temporal consistency, where CLIP-F values closer to 1 indicate better consistency. To evaluate overall transfer quality, we define the $S _ { v g g } .$ , which is computed using features extracted from VGG19 (Simonyan and Zisserman 2014). Furthermore, LPIPS and RMSE are used to measure short-term and long-term multi-view consistency (Liu et al. 2024), respectively.

Comparison Methods. We compare our method with recent state-of-the-art approaches, including VGG-based methods (StyleGaussian (Liu et al. 2024), SGSST (Galerne et al. 2025)), CLIP-based methods (CLIPGaussian (Howil et al. 2025)), and diffusion-based methods (FantasyStyle (Yang et al. 2026)).

Implementation Details. We adopt Stable Diffusion 1.5 (Rombach et al. 2022) as our base model. We fix the timestep to $t = 1$ and extract self-attention features from all blocks for loss computation. For each batch, we use $N = 4$ views and set $\lambda = \bar { 0 . 1 }$ . All experiments are conducted on a single NVIDIA L20 (48G) GPU.

<!-- image-->  
Figure 5: Qualitative comparison of different methods on scene-level datasets. Our approach achieves superior style transfer while better preserving the original content. The red boxes highlight clear differences, the details of which are further compared in Fig. 7.

<!-- image-->  
Figure 6: Qualitative comparison of different methods on object-level datasets. Other methods often suffer from over-stylization and content leakage from the style image. In contrast, our approach avoids these issues, achieving superior visual quality in style transfer.

Table 1: Quantitative comparison of different methods in 3DGS style transfer. Bold : best; underline : second best.
<table><tr><td rowspan="2">Method</td><td rowspan="2">CLIP-S â</td><td rowspan="2">CLIP-Câ</td><td rowspan="2">CLIP-CONS â</td><td rowspan="2">CLIP-F</td><td rowspan="2"> $S _ { v g g } \downarrow$ </td><td rowspan="2">FID â</td><td colspan="2">Short-range consistency RMSE</td><td colspan="2">Long-range consistency</td><td rowspan="2">Per-Instance training timeâ</td></tr><tr><td>LPPIPS </td><td></td><td>LPPIPS </td><td>RMSE</td></tr><tr><td colspan="10">Scene-level</td></tr><tr><td>StyleGaussian</td><td>0.64</td><td>0.61</td><td>0.032</td><td>1.02</td><td>25.81</td><td>334.9</td><td>0.088</td><td>0.110</td><td>0.151</td><td>0.160</td><td>â¼21min</td></tr><tr><td>SGSST</td><td>0.65</td><td>0.64</td><td>0.066</td><td>1.01</td><td>20.60</td><td>289.1</td><td>0.086</td><td>0.107</td><td>0.159</td><td>0.182</td><td>~40min</td></tr><tr><td>FantasyStyle</td><td>0.64</td><td>0.67</td><td>0084</td><td>1.01</td><td>26.88</td><td>228.6</td><td>0.08</td><td>0.101</td><td>00.167</td><td>0.173</td><td>â¼34min</td></tr><tr><td>CL PGaussian</td><td>0.79</td><td>0.63</td><td>0.058</td><td>1.03</td><td>24.68</td><td>280.7</td><td>0.081</td><td>0.074</td><td>0.161</td><td>0.127</td><td>~16min</td></tr><tr><td>Ours</td><td>0.665</td><td>0.71</td><td>0.128</td><td>1.00</td><td>20.63</td><td>204.8</td><td>0.075</td><td>0.075</td><td>0.141</td><td>0.130</td><td>~16min</td></tr><tr><td colspan="10"></td></tr><tr><td>SGSST</td><td>0.63</td><td>0.75</td><td>0.224</td><td>1.02</td><td>Object-level 13.94</td><td>333.7</td><td>0.184</td><td>0.176</td><td>0.211</td><td>0.202</td><td>â¼6min</td></tr><tr><td>FantasyStyle</td><td>0.61</td><td>0.79</td><td>0.358</td><td>1.02</td><td>18.20</td><td>265.2</td><td>0.186</td><td>0.150</td><td>0..218</td><td>0.188</td><td>~5in</td></tr><tr><td>CLPGaussian</td><td>0.72</td><td>0..79</td><td>0.504</td><td>1.03</td><td>1670</td><td>259.6</td><td>0.177</td><td>0.111</td><td>0.209</td><td>0.155</td><td>13min</td></tr><tr><td>Ours</td><td>0.63</td><td>0.84</td><td>0.412</td><td>1.01</td><td>16.44</td><td>255.8</td><td>0.176</td><td>0.119</td><td>0.208</td><td>0.161</td><td>~4min</td></tr></table>

<!-- image-->  
Figure 7: Detailed qualitative comparison of different methods. Our approach adheres more closely to the target style and avoids content leakage from the style image, compared to existing methods. Zoom in for better view.

## 5.1 Comparison Results

Quantitative Comparisons. As shown in Tab. 1, we comprehensively evaluate our method across multiple metrics covering style transfer quality, content preservation, and multi-view consistency. Overall, our method achieves the best performance across these aspects. Specifically, CLIP-Gaussian attains very high scores on CLIP-S, as it directly optimizes style transfer using CLIP-extracted features; however, this comes at the cost of inferior content preservation. Similarly, SGSST performs well on the $S _ { v g g }$ metric due to explicitly optimizing this objective. Despite not being tailored to any single metric, our method achieves competitive or superior performance across all style and content metrics. Notably, our method demonstrates significant improvements on multi-view consistency metrics, such as CLIP-CONS and LPIPS, highlighting its effectiveness in enforcing cross-view coherence. In addition, we compare the training time of different methods. Although diffusion models are substantially larger than CLIP and VGG, our method achieves training time comparable to CLIPGaussian and substantially outperforms other methods.

<!-- image-->  
Figure 8: Results of directly applying the losses defined in Eq. 4, 5 and 7 without centering and normalization. In contrast, our method achieves faster style transfer.

Qualitative Comparisons. Fig. 5, 6, and 7 present visual comparisons of different methods on both scene-level and object-level datasets. It can be clearly observed that Style-Gaussian struggles to faithfully transfer the target style and severely damages the original scene content. SGSST, based on VGG features, suffers from VGGâs limited representational capacity, making it difficult to handle complex style images and resulting in unsatisfactory stylization (e.g., 5th and 6th columns in Fig. 5). It can also produce noticeable color distortions, such as large pink and blue regions on the ground (1st and 2nd columns), and may cause content leakage from the style image (4th row in Fig. 7) or overstylization that obscures the original content (5th row in Fig. 7). FantasyStyle is able to preserve the original content relatively well; however, as it relies on IP-Adapter for style transfer, it fails to fully align with the target style. As a result, some stylized outputs deviate from the intended style appearance (e.g., the 1st, 2nd, 5th, and 6th columns in Fig. 5). CLIPGaussian, which performs stylization by aligning CLIP features, suffers from severe content leakage from the style image, as shown in Fig. 7 and Fig. 6. It often introduces explicit semantic elements from the style image, such as cat faces or human eyes, which also explains its superior performance on the CLIP-S metric. In contrast, our approach aligns with the target style more accurately than diffusionbased methods, yielding higher-quality style transfer results. Compared with VGG- and CLIP-based methods, it better preserves the original content and effectively avoids content leakage from the style image. Overall, our method achieves the best visual quality among all compared approaches.

Table 2: Quantitative results of the ablation study on the effect of Geometry-Guided Multi-View Consistency.
<table><tr><td rowspan="2">Method</td><td rowspan="2">CLIP-CONSâ</td><td rowspan="2">CLIP-F</td><td colspan="2">Short-range Consistency</td><td colspan="2">Long-range Consistency</td></tr><tr><td>LPIPS</td><td>RMSE</td><td>LPIPS </td><td>RMSE</td></tr><tr><td> $\overline { { { \bf w } / { \bf 0 } { \bf G } { \bf G } { \bf A } } }$ </td><td>0.121</td><td>1.01</td><td>0.078</td><td>0.079</td><td>0.151</td><td>0.140</td></tr><tr><td>w/o  $\mathcal { M } _ { G }$ </td><td>0.124</td><td>1.01</td><td>0.076</td><td>0.076</td><td>0.142</td><td>0.132</td></tr><tr><td>Ours</td><td>0.128</td><td>1.00</td><td>0.075</td><td>0.075</td><td>0.141</td><td>0.130</td></tr></table>

<!-- image-->  
(c) w/o â³??

<!-- image-->  
(d) Ours  
Figure 9: Qualitative results of the ablation study on Geometry-Guided Multi-View Consistency. Zoom in for better view.

## 5.2 Ablation Study

Attention-Aware Loss. We conduct an ablation study on the centering and normalization operations in the Attention-Aware Loss under the same optimization settings, with the results shown in Fig. 8. Without centering and normalization, style transfer remains incomplete, as the optimization process overemphasizes absolute feature magnitudes, resulting in slow convergence. In contrast, incorporating centering and normalization shifts the optimization focus toward feature directions, enabling faster convergence and more effective transfer of the target style.

Geometry-Guided Multi-View Consistency. We conduct extensive quantitative experiments to evaluate the effectiveness of Geometry-Guided Attention (GGA) in improving multi-view consistency, with the results summarized in Tab. 2. Our method achieves consistent improvements across all metrics, with particularly notable gains in long-term consistency. We further investigate the role of the geometryaware mask $\mathcal { M } _ { G }$ , which is designed to prevent redundant optimization over geometrically overlapping regions that could otherwise disrupt view consistency. The results show that removing $\mathcal { M } _ { G }$ results in only marginal performance degradation, since GGA already establishes strong multiview correspondences. This observation further highlights the effectiveness of GGA in enforcing multi-view consistency. The corresponding qualitative results are shown in Fig. 9. Without explicit multi-view modeling, local features of the original content tend to become blurred and overly smoothed. In contrast, we leverage geometric information to model cross-view relationships, preserving sharp local details and coherent structures across different viewpoints, thereby significantly improving visual quality.

## 6 Conclusion

In this work, we propose DiffStyle3D, a novel diffusionbased paradigm for 3DGS stylization that operates directly in the latent space, thereby avoiding unstable denoising guidance. It introduces an Attention-Aware Loss for style transfer and content preservation, and a Geometry-Guided Multi-View Consistency that injects geometric information into self-attention to form Geometry-Guided Attention, enabling cross-view correspondence modeling. Additionally, a geometry-aware mask enhances multi-view consistency by avoiding redundant optimization in overlapping regions. Extensive experiments demonstrate that DiffStyle3D achieves superior stylization quality, better visual quality, and improved content preservation compared to existing methods.

## References

Alaluf, Y.; Garibi, D.; Patashnik, O.; Averbuch-Elor, H.; and Cohen-Or, D. 2024. Cross-image attention for zero-shot appearance transfer. In ACM SIGGRAPH 2024 conference papers, 1â12.

Barron, J. T.; Mildenhall, B.; Verbin, D.; Srinivasan, P. P.; and Hedman, P. 2022. Mip-nerf 360: Unbounded antialiased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 5470â5479.

Cao, M.; Wang, X.; Qi, Z.; Shan, Y.; Qie, X.; and Zheng, Y. 2023. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of

the IEEE/CVF international conference on computer vision, 22560â22570.

Chen, M.; Laina, I.; and Vedaldi, A. 2024. Training-free layout control with cross-attention guidance. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, 5343â5353.

Chen, X.; Chu, F.-J.; Gleize, P.; Liang, K. J.; Sax, A.; Tang, H.; Wang, W.; Guo, M.; Hardin, T.; Li, X.; et al. 2025. Sam 3d: 3dfy anything in images. arXiv preprint arXiv:2511.16624.

Chen, Y.; Yuan, Q.; Li, Z.; Liu, Y.; Wang, W.; Xie, C.; Wen, X.; and Yu, Q. 2024. Upst-nerf: Universal photorealistic style transfer of neural radiance fields for 3d scene. IEEE Transactions on Visualization and Computer Graphics, 31(4): 2045â2057.

Chung, J.; Hyun, S.; and Heo, J.-P. 2024. Style injection in diffusion: A training-free approach for adapting largescale diffusion models for style transfer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 8795â8805.

Ding, S. Z.; Mao, J.; and Aizawa, K. 2024. Trainingfree sketch-guided diffusion with latent optimization. arXiv preprint arXiv:2409.00313.

Fujiwara, H.; Mukuta, Y.; and Harada, T. 2024. Stylenerf2nerf: 3d style transfer from style-aligned multi-view images. In SIGGRAPH Asia 2024 Conference Papers, 1â 10.

Galerne, B.; Wang, J.; Raad, L.; and Morel, J.-M. 2025. SGSST: Scaling Gaussian Splatting Style Transfer. In Proceedings of the Computer Vision and Pattern Recognition Conference, 26535â26544.

Gatys, L. A.; Ecker, A. S.; and Bethge, M. 2016. Image style transfer using convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2414â2423.

Gu, Z.; Li, M.; Chen, R.; Ji, Z.; Guo, S.; Zhang, Z.; Ye, G.; and Hu, Z. 2024. ArtNVG: Content-Style Separated Artistic Neighboring-View Gaussian Stylization. arXiv preprint arXiv:2412.18783.

He, F.; Li, G.; Zhang, M.; Yan, L.; Si, L.; Li, F.; and Shen, L. 2024. Freestyle: Free lunch for text-guided style transfer using diffusion models. arXiv preprint arXiv:2401.15636.

He, S.; Ji, P.; Yang, Y.; Wang, C.; Ji, J.; Wang, Y.; and Ding, H. 2025. A survey on 3d gaussian splatting applications: Segmentation, editing, and generation. arXiv preprint arXiv:2508.09977.

Heitz, E.; Vanhoey, K.; Chambon, T.; and Belcour, L. 2021. A sliced wasserstein loss for neural texture synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9412â9420.

Hertz, A.; Mokady, R.; Tenenbaum, J.; Aberman, K.; Pritch, Y.; and Cohen-Or, D. 2022. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626.

Hertz, A.; Voynov, A.; Fruchter, S.; and Cohen-Or, D. 2024. Style aligned image generation via shared attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 4775â4785.

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33: 6840â6851.

Howil, K.; Borycki, P.; Dziarmaga, T.; Mazur, M.; Spurek, P.; et al. 2025. CLIPGaussian: Universal and Multimodal Style Transfer Based on Gaussian Splatting. Advances in neural information processing systems.

Jain, S.; Kuthiala, A.; Sethi, P. S.; and Saxena, P. 2024. Stylesplat: 3d object style transfer with gaussian splatting. arXiv preprint arXiv:2407.09473.

Jing, Y.; Yang, Y.; Feng, Z.; Ye, J.; Yu, Y.; and Song, M. 2019. Neural style transfer: A review. IEEE transactions on visualization and computer graphics, 26(11): 3365â3385.

Kerbl, B.; Kopanas, G.; Leimkuhler, T.; and Drettakis, G. Â¨ 2023. 3D Gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4): 139â1.

Kovacs, Â´ A. S.; Hermosilla, P.; and Raidou, R. G. 2024. G- Â´ Style: Stylized Gaussian Splatting. In Computer Graphics Forum, volume 43, e15259. Wiley Online Library.

Lin, Y.; Lei, J.; and Jia, K. 2025. Multi-StyleGS: Stylized Gaussian Splatting with Multiple Styles. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 5289â5297.

Liu, K.; Zhan, F.; Chen, Y.; Zhang, J.; Yu, Y.; El Saddik, A.; Lu, S.; and Xing, E. P. 2023a. Stylerf: Zero-shot 3d style transfer of neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8338â8348.

Liu, K.; Zhan, F.; Xu, M.; Theobalt, C.; Shao, L.; and Lu, S. 2024. Stylegaussian: Instant 3d style transfer with gaussian splatting. In SIGGRAPH Asia.

Liu, R.; Wu, R.; Van Hoorick, B.; Tokmakov, P.; Zakharov, S.; and Vondrick, C. 2023b. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, 9298â9309.

Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.; Ramamoorthi, R.; and Ng, R. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1): 99â106.

Nichol, A. Q.; and Dhariwal, P. 2021. Improved denoising diffusion probabilistic models. In International conference on machine learning, 8162â8171. PMLR.

Podell, D.; English, Z.; Lacey, K.; Blattmann, A.; Dockhorn, T.; Muller, J.; Penna, J.; and Rombach, R. 2023. Sdxl: Im-Â¨ proving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, 8748â8763. PmLR.

Risser, E.; Wilmot, P.; and Barnes, C. 2017. Stable and controllable neural texture synthesis and style transfer using histogram losses. arXiv preprint arXiv:1701.08893.

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 10684â 10695.

Saroha, A.; Gladkova, M.; Curreli, C.; Muhle, D.; Yenamandra, T.; and Cremers, D. 2024. Gaussian splatting in style. In DAGM German Conference on Pattern Recognition, 234â 251. Springer.

Shi, Y.; Wang, P.; Ye, J.; Long, M.; Li, K.; and Yang, X. 2023. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512.

Shi, Y.; Xue, C.; Liew, J. H.; Pan, J.; Yan, H.; Zhang, W.; Tan, V. Y.; and Bai, S. 2024. Dragdiffusion: Harnessing diffusion models for interactive point-based image editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8839â8849.

Simonyan, K.; and Zisserman, A. 2014. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.

Vacher, J.; Davila, A.; Kohn, A.; and Coen-Cagli, R. 2020. Texture interpolation for probing visual perception. Advances in neural information processing systems, 33: 22146â22157.

Wang, H.; Spinelli, M.; Wang, Q.; Bai, X.; Qin, Z.; and Chen, A. 2024. Instantstyle: Free lunch towards stylepreserving in text-to-image generation. arXiv preprint arXiv:2404.02733.

Xing, P.; Wang, H.; Sun, Y.; Wang, Q.; Bai, X.; Ai, H.; Huang, R.; and Li, Z. 2024. Csgo: Content-style composition in text-to-image generation. arXiv preprint arXiv:2408.16766.

Xu, Y.; Wang, Z.; Xiao, J.; Liu, W.; and Chen, L. 2024. Freetuner: Any subject in any style with training-free diffusion. arXiv preprint arXiv:2405.14201.

Yang, Y.; Wang, Y.; Wang, C.; Wang, H.; and He, S. 2026. Fantasystyle: Controllable stylized distillation for 3d gaussian splatting. In Proceedings of the AAAI Conference on Artificial Intelligence.

Yang, Y.; Wang, Y.; Wang, C.; Zhang, Y.; Chen, Z.; and He, S. 2025a. SplitFlux: Learning to Decouple Content and Style from a Single Image. arXiv preprint arXiv:2511.15258.

Yang, Y.; Wang, Y.; Zhang, T.; Wang, J.; and He, S. 2025b. Prompt-softbox-prompt: A free-text embedding control for image editing. In Proceedings of the 33rd ACM International Conference on Multimedia, 10132â10141.

Ye, H.; Zhang, J.; Liu, S.; Han, X.; and Yang, W. 2023. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721.

Yu, X.-Y.; Yu, J.-X.; Zhou, L.-B.; Wei, Y.; and Ou, L.-L. 2024. Instantstylegaussian: Efficient art style transfer with 3d gaussian splatting. arXiv preprint arXiv:2408.04249.

Zhang, D.; Yuan, Y.-J.; Chen, Z.; Zhang, F.-L.; He, Z.; Shan, S.; and Gao, L. 2024. Stylizedgs: Controllable stylization for 3d gaussian splatting. arXiv preprint arXiv:2404.05220.

Zhang, K.; Kolkin, N.; Bi, S.; Luan, F.; Xu, Z.; Shechtman, E.; and Snavely, N. 2022. Arf: Artistic radiance fields. In European Conference on Computer Vision, 717â733. Springer.

Zhou, Y.; Gao, X.; Chen, Z.; and Huang, H. 2025. Attention distillation: A unified approach to visual characteristics transfer. In Proceedings of the Computer Vision and Pattern Recognition Conference, 18270â18280.

Zhuang, C.; Hu, Y.; Zhang, X.; Cheng, W.; Bao, J.; Liu, S.; Yang, Y.; Zeng, X.; Yu, G.; and Li, M. 2025. Styleme3d: Stylization with disentangled priors by multiple encoders on 3d gaussians. arXiv preprint arXiv:2504.15281.