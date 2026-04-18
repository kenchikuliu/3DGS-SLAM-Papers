# SSGaussian: Semantic-Aware and Structure-Preserving

3D Style Transfer

Jimin Xu, Bosheng Qin, Tao Jin, Zhou Zhao, Zhenhui Ye, Jun Yu, Fei Wu, Senior Member, IEEE

AbstractâRecent advancements in neural representations, such as Neural Radiance Fields and 3D Gaussian Splatting, have increased interest in applying style transfer to 3D scenes. While existing methods can transfer style patterns onto 3Dconsistent neural representations, they struggle to effectively extract and transfer high-level style semantics from the reference style image. Additionally, the stylized results often lack structural clarity and separation, making it difficult to distinguish between different instances or objects within the 3D scene. To address these limitations, we propose a novel 3D style transfer pipeline that effectively integrates prior knowledge from pretrained 2D diffusion models. Our pipeline consists of two key stages: First, we leverage diffusion priors to generate stylized renderings of key viewpoints. Then, we transfer the stylized key views onto the 3D representation. This process incorporates two innovative designs. The first is cross-view style alignment, which inserts cross-view attention into the last upsampling block of the UNet, allowing feature interactions across multiple key views. This ensures that the diffusion model generates stylized key views that maintain both style fidelity and instance-level consistency. The second is instance-level style transfer, which effectively leverages instance-level consistency across stylized key views and transfers it onto the 3D representation. This results in a more structured, visually coherent, and artistically enriched stylization. Extensive qualitative and quantitative experiments demonstrate that our 3D style transfer pipeline significantly outperforms state-of-theart methods across a wide range of scenes, from forward-facing to challenging 360-degree environments. Visit our project page https://jm-xu.github.io/SSGaussian/ for immersive visualization.

Index Termsâ3D Style Transfer, 3D Gaussian Splatting, Diffusion.

## I. INTRODUCTION

R ECENT advancements in diffusion models have signifi-cantly improved image and video generation [1]â[6]. In cantl prove magen ideo eneration [[. In the 3D domain, diffusion models have also been integrated with neural representations, such as Neural Radiance Fields (NeRF) [7] and 3D Gaussian Splatting (3DGS) [8], effectively lifting 2D priors into the 3D world. DreamFusion [9] and its follow-up works [10], [11] introduce Score Distillation Sampling (SDS) loss, which is based on probability density distillation. Through SDS loss, 2D prior knowledge can be utilized to guide 3D content generation. Similarly, Instruct-NeRF2NeRF [12] and related approaches [13], [14] introduce the Iterative Dataset Update (IDU) method, which employs InstructPix2Pix [15] to iteratively edit rendered images while updating the underlying 3D representations. These advancements have led to remarkable progress in 3D generation and editing. However, in the field of 3D style transfer, there still lack effective methods that integrate diffusion priors into a systematically designed 3D style transfer pipeline.

Given a reconstructed 3D scene and a reference style image, 3D style transfer aims to convert the 3D scene into the reference style while preserving the original content structure. Early 3D style transfer methods were based on point cloud representations [16], [17] or mesh representations [18], [19]. However, these approaches often produce noticeable artifacts when applied to complex real-world scenes, mainly due to imperfections in geometry reconstruction and texture rendering [20]. More recently, neural representations such as NeRF [7] and 3DGS [8] have gained significant attention in 3D scene reconstruction. As these neural representations have demonstrated superior fidelity and flexibility, they have also been increasingly adopted in 3D style transfer tasks, replacing traditional point cloud and mesh-based approaches. Current 3D style transfer methods based on NeRF and 3DGS can be broadly categorized into feed-forward approaches [21], [22] and iterative optimization approaches [20], [23]. However, both approaches struggle to effectively extract and transfer style semantics from the reference image. Furthermore, the stylized results often lack a layered sense of structure, making it difficult to distinguish between different instances or objects within the 3D scene.

To address these limitations, we propose SSGaussian, a 3D style transfer pipeline that effectively exploits 2D diffusion priors. First, we reconstruct a 3D scene representation using Gaussian Grouping [24], which extends 3DGS to jointly reconstruct and segment anything in 3D scenes. Once the reconstruction is complete, we select key viewpoints and render their corresponding RGB images and depth maps. Given these key view renderings along with a style image prompt, we leverage a pretrained diffusion model to generate stylized outputs. Diffusion models excel at transferring style semantics while preserving content and structural integrity. However, ensuring multi-view consistency remains a significant challenge, as existing 2D diffusion models struggle to maintain coherence across different viewpoints. Since the consistency of these stylized key views is crucial for high-quality 3D style transfer, we introduce a Cross-View Style Alignment (CVSA) module to address this issue. Instead of enforcing pixel-level 3D consistency, which is inherently difficult, our module focuses on instance-level consistencyâensuring that the same objects or instances across different key views retain a uniform stylization. To achieve this, we incorporate cross-view attention into the last upsampling block of the UNet [25], allowing feature interactions across multiple key views. Accompanied with our CVSA module, the diffusion model generates stylized key views that maintain both style fidelity and instance-level consistency, establishing a solid foundation for the subsequent 3D style transfer process.

<!-- image-->  
Fig. 1. Pipeline of SSGaussian. We begin by reconstructing the scene using a 3D Gaussian Splatting representation. Next, we select key viewpoints and render their corresponding RGB images. Then, given a reference style image, we apply a pretrained diffusion model enhanced with our proposed Cross-View Style Alignment module to generate consistent stylized results for the key views. Finally, we achieve full 3DGS stylization by transferring the stylized key views onto the 3D representation through our Instance-level Style Transfer approach.

We further introduce a novel 3D Gaussians stylization algorithm built upon group matching, which effectively transfers stylized key views onto the 3D representation. To fully leverage the instance-level consistency across stylized key views, our key insight is to utilize instance segmentation to establish correspondences between local regions in the training view and their counterparts in the key views. Within each matched local region, we perform nearest-neighbor search and minimize the distance between corresponding features. Specifically, by leveraging the Identity Encoding parameters introduced in Gaussian Grouping [24], we can obtain group identities for different local regions in any given view. Each group corresponds to a distinct instance in the 3D scene. Using these group identities, we match each group in the training view to the group with the same identity in the stylized key views, ensuring that the same instance is consistently associated across different viewpoints. Building upon this group matching mechanism, we propose an Instancelevel Style Transfer (IST) approach that enables localized and semantically coherent stylization. For every sampled training view, the objective is to minimize the cosine distance between each feature in a group and its closest counterpart in the stylized key views. By incorporating this approach, our 3D style transfer method could better preserve both high-level style semantics and fine-grained brushstroke details, which facilitates a more structured, visually coherent, and artistically enriched stylization in the final rendered outputs.

We conduct experiments on a variety of scenes, ranging from forward-facing scenes to challenging 360-degree scenes. To ensure a comprehensive comparison, we select a diverse set of style reference images, enabling a thorough evaluation of the stylization performance of our 3D style transfer pipeline against baseline methods. Additionally, we perform ablation studies on different components of our pipeline to validate their effectiveness. Both qualitative and quantitative results demonstrate that our 3D style transfer pipeline significantly outperforms state-of-the-art methods.

We summarize our contributions as follows:

â¢ We propose a novel 3D style transfer pipeline, which effectively integrates diffusion priors, achieving highquality and visually coherent style transfer results.

â¢ To address the multi-view consistency challenges of 2D diffusion models, we introduce a Cross-View Style Alignment module, which ensures instance-level consistency across different viewpoints.

â¢ We propose an Instance-level Style Transfer approach with a group matching mechanism, which effectively lifts 2D diffusion priors into 3D stylization.

Qualitative and quantitative experiments demonstrate that our 3D style transfer pipeline significantly outperforms state-of-the-art methods on a variety of scenes.

## II. RELATED WORK

Diffusion Model-based Neural Style Transfer. Neural style transfer [26]â[33] aims at transferring the style of a reference image to the target content image. In the field of text-to-image diffusion models [5], recent studies have shown that style transfer can be achieved by manipulating the diffusion process through self-attention or cross-attention mechanisms [6], [34]â [38]. For example, InstantStyle [38] and Style-Adapter [35] utilize lightweight adapters to extract image features and inject them into cross-attention layers. StyleAlign [36] and StyleID [37] perform style transfer by swapping the key and value features of the self-attention block with those from a reference style image.

Text-driven 3D Editing. With the growing popularity of textto-image diffusion models [5], researchers are exploring ways to leverage these models for editing 3D scenes based on text instructions. Instruct-NeRF2NeRF (IN2N) [12] introduces the Iterative Dataset Update optimization algorithm, effectively transforming the 3D NeRF [7] editing challenge into a 2D image editing task. It then employs InstructPix2Pix (IP2P) [15] to enable instruction-based 2D image editing. ViCA-NeRF [39] offers a more flexible and efficient editing approach by editing key views and incorporating a blending refinement model, ensuring consistent edits without requiring iterative updates in NeRF. DreamEditor [40] converts NeRF representations into meshes and directly optimizes the mesh using score distillation sampling (SDS) loss [9]. GaussianEditor [13] enhances 3D editing by leveraging the explicit representation properties of Gaussian Splatting [8]. GaussCtrl [41] focuses on designing guidance mechanisms within diffusion models, leading to faster editing and improved visual quality. Other works [42] have explored the use of CLIP [43] for Text-driven 3D Editing. 3D Style Transfer. 3D style transfer aims to transform a 3D scene so that renderings from different viewpoints align with the style of a target image while preserving the original content structure. Previous research suggests that stylizing a 3D scene can be explicitly achieved using point cloud [16], [17] and mesh representations [18], [19]. However, such approaches often produce noticeable artifacts in complex real-world scenes due to imperfections in geometry and texture rendering. [20] Recently, NeRF [7] and Gaussian Splatting [8] have gained significant prominence in 3D scene reconstruction, demonstrating a superior ability to accurately reproduce the appearance of real-world scenes. Consequently, these techniques have also emerged as the dominant 3D representations for 3D style transfer.

<!-- image-->  
Fig. 2. Two Stage Stylization. We decompose the 3D style transfer task into two sequential stages: the stylization of key views and the stylization of the 3D Gaussian Splatting (3DGS) representation based on those stylized key views. In Stage 1, given a style reference image along with RGB and depth images rendered from the 3DGS, we design a diffusion model to effectively transfer style semantics to the selected key viewpoints. In Stage 2, leveraging group matching between the key viewpoints and training views, we introduce an Instance-level Style Transfer approach that hierarchically transfers the style semantics onto the 3DGS representation.

One line of approaches follows a feed-forward manner [21], [22]. These methods require training a neural network for each 3D scene representation, enabling the stylization of the 3D scene in a single forward pass during inference. However, these approaches are limited by the style domain of the training set and often struggle to accurately reproduce fine style patterns, such as colors and brushstrokes. Another line of approaches performs 3D style transfer through iterative optimization [14], [20], [23], [44], minimizing content and style losses. ARF [20] introduces a novel loss function based on nearest neighbor feature matching (NNFM), which better preserves fine details from the style images. LSNeRF [44] proposes a spatial matching mechanism between the style image and NeRF renderings, allowing dynamically assigned correspondences to guide the stylization process and produce diverse results. InstantStyleGaussian [14] incorporates the Iterative Dataset Update optimization algorithm from GaussianEditor [13] into 3D style transfer. G-Style [23] introduces a dual-loss function that enables the algorithm to capture both high-frequency and low-frequency patterns in the style image, enhancing the overall stylization quality.

## III. METHOD

We propose a novel pipeline for 3D style transfer that effectively transfers both large-scale style semantics and finescale style patterns (such as brushstrokes), resulting in a more structured and visually coherent stylized rendering. As illustrated in Figure 1, our pipeline consists of the following key steps:

Scene Reconstruction. Reconstruct the scene using a 3D Gaussian Splatting (3DGS) [8] representation.

Key View Selection and Rendering. Select key viewpoints and render their corresponding RGB images.

<!-- image-->  
Fig. 3. Impact of the Cross-View Style Alignment module across different blocks of the denoising U-Net.

Consistent Multi-view Stylization. Apply a pretrained diffusion model [5] equipped with our proposed Cross-View Style Alignment module to generate consistent stylized key views. 3DGS Stylization. Achieve 3DGS stylization with these stylized key views by leveraging our Instance-level Style Transfer approach.

Our 3D style transfer pipeline can be decomposed into two sequential stages: stylization of key views and stylization of the 3DGS representation based on the stylized key views.

## A. Preliminaries: 3D Gaussian Grouping

3D Gaussian Splatting [8] represents a 3D scene as a collection of 3D colored Gaussians. Each Gaussian is defined by several properties: a mean $\mu ~ \in ~ \mathbb { R } ^ { 3 }$ that determines its center, a covariance matrix $\Sigma \in \mathbb { R } ^ { 3 \times 3 }$ that characterizes its shape and size, an opacity value $\alpha \in \mathbb { R }$ , and a color vector $c .$ Gaussian Splatting projects these 3D Gaussians onto the 2D image plane, and the color $C$ of a pixel is rendered by blending N ordered Gaussians that overlap with the pixel:

$$
C = \sum _ { i \in N } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } \left( 1 - \alpha _ { j } \right) .\tag{1}
$$

Compared to NeRF, 3DGS offers remarkable reconstruction quality with faster rendering speeds. To enable 3D Gaussians for fine-grained scene understanding, Gaussian Grouping [24] assigns each Gaussian to its corresponding instances or stuff within the 3D scene. This method first deploys Segment Anything Model (SAM) [45] to automatically generate masks for each view independently. It then utilizes a universal temporal propagation model [46] to associate mask labels across views, producing a coherent multi-view segmentation as training input. Gaussian Grouping introduces new Identity Encoding parameters, which are integrated into the properties of each 3D Gaussian and jointly optimized during training. The Identity Encoding parameters follow a format similar to color modeling and share a comparable rendering process.

## B. Consistent Multi-view Stylization.

As illustrated in Figure 2 (Stage 1), we leverage the prior knowledge of a pretrained diffusion model to transfer style semantics to key viewpoints. To enable the diffusion model to support style image prompts, we integrate IP-Adapter [6], which introduces an image encoder to extract features from the style image prompt. A small projection network consisting of a linear layer and Layer Normalization (LN) [47] projects the image embedding into a feature sequence. These features are then embedded into the pretrained diffusion model through adapted modules with decoupled cross-attention.

To ensure that the generated stylized images preserve the content information of the original views, we employ ControlNet [48] to regulate the generation process. Benefiting from the 3DGS reconstruction, we can extract consistent depth maps across key views. Using a depth-conditioned ControlNet facilitates the generation of multi-view consistent stylized images.

Given an input image I to be stylized, a style image prompt $P ,$ and a depth map D, the diffusion model first encodes the input image I into a latent code $z _ { \mathrm { 0 } }$ using a VAE encoder. It then iteratively adds noise to the latent code. For this diffusion process, we adopt DDIM inversion [49], which has been demonstrated by GaussCtrl [41] to obtain consistent initial noise across multiple views for stable stylization. The DDIM inversion process is formulated as follows:

$$
z _ { t } = \frac { \sqrt { \alpha _ { t } } } { \sqrt { \alpha _ { t - 1 } } } \left( z _ { t - 1 } - \sqrt { 1 - \alpha _ { t - 1 } } \epsilon _ { t } \right) + \sqrt { 1 - \alpha _ { t } } \epsilon _ { t } ,\tag{2}
$$

where t is the time step of the diffusion process, $\epsilon _ { t }$ is the noise predicted by the UNet, and $\alpha _ { t }$ is the scheduling coefficient in DDIM scheduler. After T steps of the diffusion process, we obtain a noisy latent representation $z _ { T } .$ . The stylized latent representation $z _ { 0 } ^ { \prime }$ is then recovered using the DDIM denoising process:

$$
z _ { t - 1 } ^ { \prime } = \sqrt { \alpha _ { t - 1 } } \left( \frac { z _ { t } ^ { \prime } - \sqrt { 1 - \alpha _ { t } } \epsilon _ { t } } { \sqrt { \alpha _ { t } } } \right) + \sqrt { 1 - \alpha _ { t - 1 } } \epsilon _ { t } ,\tag{3}
$$

To further ensure instance-level consistency across stylized key viewsâi.e., maintaining consistent stylization for the same instances or objects across different key views of the 3D sceneâwe design a Cross-View Style Alignment Module. This module inserts cross-view attention at the last upsampling block of the UNet. Before diving into our solution, we revisit the concept of self-attention [50] within diffusion models. The self-attention layer takes image features $\boldsymbol { z } \in \mathbb { R } ^ { H W \times d _ { i } }$ as input and computes attention as follows:

$$
\mathrm { A t t n } ( z , z ) = \mathrm { S o f t m a x } ( \frac { Q \left( z \right) K ( z ) ^ { T } } { \sqrt { d } } ) V ( z ) ,\tag{4}
$$

where $Q , K , V$ are linear transformations used to obtain image queries, keys and values $Q \left( z \right) , K ( z ) , V ( z ) \in \mathbb { R } ^ { H W \times d }$

This mechanism enables information exchange across spatial locations within the same view. However, to achieve multi-view consistency, we extend this formulation by introducing cross-view attention, allowing feature interactions across different key views. Specifically, we compute crossview attention by allowing the query matrix to attend to the key and value matrices of other key views $z ^ { 1 : K }$

<!-- image-->  
Fig. 4. Qualitative comparison on LLFF dataset. We compare our SSGaussian against the state-of-the-art methods on the task of stylizing forward-facing scenes from a reference style image.

$$
\mathrm { A t t n } ( z , z ^ { 1 : K } ) = \mathrm { S o f t m a x } ( \frac { Q \left( z \right) K ( z ^ { 1 : K } ) ^ { T } } { \sqrt { d } } ) V ( z ^ { 1 : K } ) ,\tag{5}
$$

Through empirical analysis, we observe that injecting Cross-View Style Alignment at different blocks of the UNet yields varying degrees of multi-view consistency. As shown in Figure 3, placing the cross-view attention in the early blocks often leads to insufficient semantic alignment across views, such as the material of the truckâs back bucket and tire. In contrast, inserting the module at the last upsampling block consistently achieves the best trade-off between preserving fine-grained style details and maintaining cross-view semantic consistency. We attribute this to the fact that features at the last upsampling stage are semantically rich and spatially refined, making them well-suited for enforcing instance-level consistency across views without disrupting the overall generative fidelity.

This process ensures that the generated stylized images maintain both style fidelity and multi-view consistency, laying a solid foundation for the subsequent 3D style transfer stage.

## C. 3D Gaussians Stylization

Given the stylized images of key views S, our goal is to transfer the stylization to the 3DGS representation, enabling novel view synthesis with consistent stylization. However, since the stylized key views generated by diffusion model lack strict 3D consistency constraints, they can only ensure instance-level consistency to some extent but do not guarantee pixel-level 3D consistency. As a result, directly fine-tuning 3DGS using these stylized key views often leads to errorprone optimization, where the stylized novel views tend to become blurry and exhibit artifacts. To better utilize the style information from the key views in guiding the fine-tuning of 3DGS, we propose a 3D Gaussians stylization algorithm build upon a group matching mechanism, as illustrated in Figure 2 (Stage 2). This approach enhances the transfer of both style semantics and brushstroke details, leading to a more structured and visually coherent stylization in the final rendered results. Group Matching We first match the local regions of the sampled training view I with those of the stylized key views S. Leveraging the Identity Encoding parameters introduced by Gaussian Grouping [24], we can obtain the group identity for different local regions in any given view, where each group represents a specific instance in the 3D scene. Specifically, Gaussian Grouping introduces new Identity Encoding parameters to each 3D Gaussian. We denote the Identity Encoding as $e _ { i } ,$ which is a learnable and compact vector of length 16. This encoding is sufficient to distinguish different objects or parts within the scene while maintaining computational efficiency. Similar to equation 1, we can determine the group identity for each pixel in any given view as follow:

<!-- image-->  
Fig. 5. Qualitative comparison on Tanks and Temples dataset. We compare our SSGaussian against the state-of-the-art methods on the task of stylizing 360-degree scenes from a reference style image.

$$
E _ { i d } = \sum _ { i \in { \cal N } } e _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } \left( 1 - \alpha _ { j } \right) .\tag{6}
$$

Next, we apply a linear layer $f$ to restore the feature dimension back to K and then apply a softmax function $s o f t m a x ( f ( E _ { i d } ) )$ for identity classification, where K represents the total number of groups in the 3D scene. Through this process, we obtain the group identity map for any given view. We denote the group identity map of the sampled training view as $M _ { I }$ and that of the stylized key views as $M _ { S }$ . Using the group identity map, we aggregate positions with the same group identity into distinct groups. Let the group sets of the sampled training view and the stylized key views be denoted as $\bar { X } = \{ x _ { i } \} _ { i = 1 } ^ { K }$ and $Y = \{ y _ { i } \} _ { i = 1 } ^ { \dot { K } }$ , respectively. To provide style guidance for the sampled training view, we match its groups with those of the stylized key views. This matching process can be formulated as constructing a mapping M:

$$
\mathcal { M } ( x _ { i } ) = \left\{ \begin{array} { l l } { y _ { i } , } & { y _ { i } \neq \emptyset , } \\ { \bigcup _ { j = 1 } ^ { K } y _ { j } , } & { y _ { i } = \emptyset , } \end{array} \right.\tag{7}
$$

The mapping matrix M maps the group with identity i in the sampled training view to the group with the same identity i in the stylized key view, thereby associating the same instance across different views. If there is no region with identity i in the stylized key view, i.e., $y _ { i } = \varnothing$ , it is linked to the global region, where the most appropriate style is matched globally. Instance-level Style Transfer. Given the sampled training view I, the stylized key views S, and their corresponding group matching mapping M, we propose an Instancelevel Style Transfer (IST) approach, which performs nearestneighbor feature matching (NNFM) within local groups. We extract the high-level VGG feature maps $F _ { I }$ and $F _ { S }$ for I and $S ,$ respectively. Let $F _ { I } ( i , j )$ denote the feature vector at pixel location $( i , j )$ in the feature map $F _ { I }$ . Our style loss is formulated as:

$$
\mathcal { L } _ { S } = \frac { 1 } { N } \sum _ { x _ { k } \in X } \sum _ { ( i , j ) \in x _ { k } } \operatorname* { m i n } _ { ( i ^ { \prime } , j ^ { \prime } ) \in \mathcal { M } ( x _ { k } ) } D \left( F _ { I } ( i , j ) , F _ { S } ( i ^ { \prime } , j ^ { \prime } ) \right) ,\tag{8}
$$

where N is the number of pixels in $F _ { I }$ , and D computes the cosine distance between two vectors. Each group in the view can be considered as an instance or stuff within the 3D scene. Our IST approach performs localized nearest-neighbor feature matching between each group in the sampled training view and its corresponding group in the stylized key views. The objective is to minimize the cosine distance between each feature within a group and its nearest neighbor within the feature space of the matching stylized key views.

## IV. EXPERIMENTS

## A. Datasets

We conduct experiments on two real-world scene datasets: the LLFF dataset [51] and the Tanks and Temples dataset [52]. The LLFF dataset consists of forward-facing scenes with minimal variation in camera pose. It includes real-world scenes with complex geometric structures; for example, the Fern scene contains highly detailed areas such as leaves, which present a challenge for stylization. The Tanks and Temples dataset features 360-degree scenes captured in unbounded real-world environments. Unlike LLFF, it includes significant variations in camera poses, making it particularly suitable for evaluating the consistency and robustness of 3D stylization methods.

## B. Comparisons with the State of the Arts

Baselines. We compare our approach with three state-ofthe-art methods: Artistic Radiance Fields (ARF) [20], an iterative optimization approach for NeRF scenes. StyleGaussian [22], a feed-forward approach designed for 3DGS scenes. G-Style [23], a recent iterative optimization approach applied to 3DGS scenes.

Qualitative comparison. We compare our method against three representative baseline approaches on both the LLFF and Tanks and Temples datasets. In Figure 4, we present four forward-facing scenes from the LLFF dataset: Trex, Fern, Horns, and Flower. In Figure 5, we showcase two 360-degree scenes from the Tanks and Temples dataset: Train and Truck. For each 360-degree scene, we sample two distant views to display the stylized renderings, allowing for a qualitative comparison of multi-view stylization consistency. To ensure a comprehensive evaluation, we apply a diverse set of style reference images in Figure 4 and Figure 5, including abstract art, sketches, cartoons, fantasy art, oil paintings, and ink wash paintings. This variety enables a more thorough comparison of the stylization results across different baselines.

From the qualitative comparisons in Figure 4 and Figure 5, we conclude that our method significantly outperforms the baselines in both large-scale style semantics and fine-grained style details. At a large scale, our approach effectively integrates diffusion model priors, allowing it to incorporate the semantic information from the reference style image while better preserving the content and structure of the original views. For instance, in Figure 4, for highly detailed scenes like Fern and Flower, which contain intricate leaf structures, our 3D style transfer method maintains these fine details more effectively than the baselines. Furthermore, by employing our group matching method, we enable localized stylization for different regions within the 3D scene, leading to a more structured and visually coherent stylization effect. This results in better distinction between different regions, as demonstrated in the stylized Fern and Flower scenes, where the primary subjects are more clearly differentiated. At a small scale, our method also excels in transferring fine stylistic details such as brushstrokes and object contours more accurately than the baselines. For example, in the Truck scene, the stylized lines of the truck and the brushstroke details of the trees align more closely with the reference style image, further highlighting the superiority of our approach in fine-grained stylization.

Quantitative comparison. 3D style transfer is a relatively new and under-explored research area. And currently, there still does not exist a standard quantitative metric for evaluating stylization quality. Following previous work [21], [22], we focus on assessing multi-view stylization consistency. Specifically, we measure short-range consistency between two adjacent views, and long-range consistency between two distant views. For any two given views, we warp one view to the other based on optical flow [53] using softmax splatting [54], and then compute the masked RMSE score and LPIPS score [55] to quantify the consistency of the stylization. As shown in Table I, our approach significantly outperforms baseline methods across both metrics, demonstrating superior stylization

TABLE I  
QUANTITATIVE RESULTS. WE EVALUATE THE PERFORMANCE OF SSGAUSSIAN AGAINST STATE-OF-THE-ART METHODS IN TERMS OF SHORT-RANGE CONSISTENCY AND LONG-RANGE CONSISTENCY, USING LPIPS (â) AND RMSE (â).
<table><tr><td rowspan="2">Method</td><td colspan="2">Short-range Consistency</td><td colspan="2">Long-range Consistency</td></tr><tr><td>LPIPS</td><td>RMSE</td><td>LPIPS</td><td>RMSE</td></tr><tr><td>ARF [20]</td><td>0.049</td><td>0.041</td><td>0.128</td><td>0.082</td></tr><tr><td>StyleGaussian [22]</td><td>0.036</td><td>0.030</td><td>0.077</td><td>0.071</td></tr><tr><td>G-Style [23]</td><td>0.035</td><td>0.035</td><td>0.089</td><td>0.072</td></tr><tr><td>Ours</td><td>0.031</td><td>0.028</td><td>0.073</td><td>0.068</td></tr></table>

TABLE II

QUANTITATIVE RESULTS. WE EVALUATE THE PERFORMANCE OF SSGAUSSIAN AGAINST STATE-OF-THE-ART METHODS IN TERMS OF RENDERINGS QUALITY, USING CONTENT LOSS (â) AND STYLE LOSS (â).
<table><tr><td>Method</td><td>Content Loss</td><td>Style Loss</td></tr><tr><td>ARF [20]</td><td>2.490</td><td>3.184</td></tr><tr><td>StyleGaussian [22]</td><td>2.300</td><td>5.297</td></tr><tr><td>G-Style [23]</td><td>2.467</td><td>3.303</td></tr><tr><td>Ours</td><td>2.298</td><td>3.091</td></tr></table>

consistency.

Furthermore, to evaluate the quality of stylized renderings, we employ two loss functions commonly used in image style transfer: the Gram Matrix Loss (Style Loss) and the Feature Reconstruction Loss (Content Loss) [31]. Specifically, the Style Loss quantifies the discrepancy between novel views rendered from stylized 3D Gaussians and the style reference image. This is computed as the difference between Gram matrices of feature maps extracted from specific layers of a pretrained VGG network. Lower values indicate greater similarity in style statistics captured at the selected feature layers. The Content Loss measures the mean squared error (MSE) between feature maps of stylized novel views and their original (non-stylized) counterparts at corresponding viewpoints, again using a pretrained VGG network. Lower MSE values correspond to better preservation of content structure information in the chosen feature layers. As shown in Table II, our approach outperforms baseline methods across both metrics, demonstrating superior stylization quality.

Speed comparison. We measured the training duration and rendering speed of all compared methods using a single NVIDIA RTX 3090 GPU. As shown in Table III, our approach achieves efficient stylization and real-time rendering performance comparable to the fastest alternative. Our approach, along with ARF [20] and G-Style [23], employs iterative optimization for 3D style transfer. Specifically, during each training iteration, these methods directly optimize the 3D scene parameters based on the reference style image. In contrast, StyleGaussian [22] adopts a feed-forward approach that requires training a dedicated 3D CNN decoder for each 3D scene representation. This enables single-pass stylization during inference. For the evaluated scenes: Our method completes consistent multi-view stylization in 1 minute and 3D Gaussian stylization in 19 minutes, achieving a rendering speed of 118 FPS. ARF [20] requires 24 minutes for stylization and, being NeRF-based, renders at 10 FPS. G-Style [23] consumes 9 minutes in preprocessing and 22 minutes in stylization, rendering at 110 FPS. StyleGaussian [22] necessitates approximately 5 hours per scene for CNN decoder training, with style transfer operating at 3 FPS.

<!-- image-->  
Fig. 6. Comparisons with Video Style Transfer Methods.

User study. To further assess the quality of our approach, we conduct a user study involving 30 participants. Each participant is presented with stylization results from our method, ARF [20], StyleGaussian [22], and G-Style [23] alongside their corresponding style exemplars and original content scenes. Without disclosing methodological information, participants evaluate outputs based on three criteria:

â¢ Structural Integrity: Participants are instructed to select the stylization output that best preserve the core structural elements of the original scenes (e.g., object instances, spatial layout, and edge coherence).

â¢ Style Similarity: Participants identify outputs that most faithfully match the artistic attributes of the style exemplars, encompassing low-level features (e.g., brushstroke patterns, color palette) and high-level style semantics.

â¢ Visual Quality: Participants evaluate outputs based on overall perceptual excellence, considering artifact minimization (e.g., blurring, distortions), multi-view consistency, and aesthetic appeal.

As shown in Table IV, our approach outperforms all comparative methods across every evaluation dimension, demonstrating superior performance in semantic-aware and structurepreserving 3D style transfer.

## C. Comparisons with Video Style Transfer Methods

We compare our SSGaussian against video-based style transfer methods AnyV2V [56] and UniVST [57], reformulating the task as temporal stylization by treating multi-view image sequences as video inputs. As evidenced in Figure 6, AnyV2V [56] exhibits temporal inconsistency across frames and structural degradation of scene content. UniVST [57] demonstrates inferior style transfer fidelity compared to our approach. SSGaussian achieves superior performance in both style consistency and structural preservation.

TABLE III  
SPEED COMPARISON. SSGAUSSIAN ACHIEVES EFFICIENT STYLIZATION AND REAL-TIME RENDERING PERFORMANCE COMPARABLE TO THE FASTEST ALTERNATIVE.
<table><tr><td>Method</td><td>Training Time</td><td>Rendering Speed</td></tr><tr><td>ARF [20]</td><td>24 mins</td><td>10 FPS</td></tr><tr><td>StyleGaussian [22]</td><td>5 hours</td><td>3 FPS</td></tr><tr><td>G-Style [23]</td><td>31 mins</td><td>110 FPS</td></tr><tr><td>Ours</td><td>20 mins</td><td>118 FPS</td></tr></table>

TABLE IV

USER STUDY. THE REPORTED VALUES INDICATE THE PERCENTAGE PREFERENCE FOR EACH METHOD.
<table><tr><td>Method</td><td>Structural Integrity</td><td>Style Similarity</td><td>Visual Quality</td></tr><tr><td>ARF [20]</td><td>30.0 %</td><td>20.0 %</td><td>26.7 %</td></tr><tr><td>StyleGaussian [22]</td><td>23.3 %</td><td>0.0 %</td><td>3.3 %</td></tr><tr><td>G-Style [23]</td><td>10.0 %</td><td>26.7 %</td><td>13.3 %</td></tr><tr><td>Ours</td><td>36.7 %</td><td>53.3 %</td><td>56.7 %</td></tr></table>

## D. Ablation Studies

Ablation on Cross-View Style Alignment. We perform ablation experiments to evaluate the impact of Cross-View Style Alignment (CVSA) module, as illustrated in Figure 7. The results demonstrate that our CVSA module significantly improves multi-view consistency in both large-scale style semantics and fine-grained style details. For instance, in Figure 7 (B), CVSA module effectively maintains the color and wooden material of the truckâs back bucket. Similarly, in Figure 7 (A), it ensures consistent style transfer across views. Notably, Our CVSA module does not compromise object details: both ablation methods successfully synthesize fine-grained features, such as the label on the truck cabin in Figure 7 (B).

Ablation on Instance-level Style Transfer. We conduct an ablation study to evaluate the effectiveness of Instance-level Style Transfer (IST), as illustrated in Figure 8. We compare our IST approach with the direct fine-tuning approach (w/o IST) in the second stage. In the direct fine-tuning approach, the stylized key views are used to replace the original training views, and the 3DGS is fine-tuned until convergence on these stylized key views. The quality of this baseline heavily depends on the 3D consistency of the stylized key views. However, since the stylized key views generated by the diffusion model lack strict 3D consistency constraints, they can only maintain instance-level consistency to some extent but do not guarantee pixel-level 3D consistency. As shown in Figure 8, this direct fine-tuning approach often results in error-prone optimization, where stylized novel views tend to become blurry and exhibit artifacts. For instance, in Figure 8 (A), artifacts occur at the boundaries of the background. And in Figure 8 (B), the strokes and contours of the truckâs back bucket appear blurry. In contrast, our IST approach achieves high-quality 3D style transfer, effectively transferring strokes and preserving the sharpness of object contours. Additionally, it maintains clear hierarchical distinctions between different instances within the scene, resulting in a more coherent and

with CVSA  
w/o CVSA  
<!-- image-->  
Fig. 7. Ablation experiment on the proposed Cross-View Style Alignment (CVSA). (A) and (B) show different scenes, with two views rendered for each. Our CVSA module significantly improves multi-view consistency in both large-scale style semantics and fine-grained style details.

with IST  
w/o IST  
<!-- image-->  
Fig. 8. Ablation experiment on the proposed Instance-level Style Transfer (IST). (A) and (B) show different scenes, with two views rendered for each. Our IST approach enables high-quality 3D style transfer by effectively reducing blurriness and mitigating visual artifacts.

visually appealing stylization.

## V. CONCLUSION

In this paper, we present a novel 3D style transfer pipeline that effectively integrates prior knowledge from pretrained 2D diffusion models to extract and transfer high-level style semantics from a reference style image. Our pipeline enables the faithful and consistent transfer of these style semantics onto 3D Gaussian Splatting representations. To ensure that the diffusion model generates stylized key views with both style fidelity and instance-level consistency, we introduce a Cross-View Style Alignment module, which incorporates crossview attention into the last upsampling block of the UNet, facilitating feature interactions across multiple key viewpoints. Additionally, we propose an Instance-level Style Transfer approach, which exploits instance-level consistency across stylized key views and transfers it onto the 3D representation in a semantically structured manner. Extensive experiments across a wide variety of scenesâincluding forward-facing and challenging 360-degree environmentsâdemonstrate the superiority of our 3D style transfer pipeline over existing methods.

## REFERENCES

[1] Y. Xu, X. Xu, H. Gao, and F. Xiao, âSgdm: An adaptive styleguided diffusion model for personalized text to image generation,â IEEE Transactions on Multimedia, vol. 26, pp. 9804â9813, 2024.

[2] C. Zhang, W. Yang, X. Li, and H. Han, âMmginpainting: Multi-modality guided image inpainting based on diffusion models,â IEEE Transactions on Multimedia, vol. 26, pp. 8811â8823, 2024.

[3] Y. Jiang, Q. Liu, D. Chen, L. Yuan, and Y. Fu, âAnimediff: Customized image generation of anime characters using diffusion model,â IEEE Transactions on Multimedia, vol. 26, pp. 10 559â10 572, 2024.

[4] H. Chen, X. Wang, G. Zeng, Y. Zhang, Y. Zhou, F. Han, Y. Wu, and W. Zhu, âVideodreamer: Customized multi-subject text-to-video generation with disen-mix finetuning on language-video foundation models,â IEEE Transactions on Multimedia, vol. 27, pp. 2875â2885, 2025.

[5] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, âHighresolution image synthesis with latent diffusion models,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684â10 695.

[6] H. Ye, J. Zhang, S. Liu, X. Han, and W. Yang, âIp-adapter: Text compatible image prompt adapter for text-to-image diffusion models,â arXiv preprint arXiv:2308.06721, 2023.

[7] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[8] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[9] B. Poole, A. Jain, J. T. Barron, and B. Mildenhall, âDreamfusion: Textto-3d using 2d diffusion,â in The Eleventh International Conference on Learning Representations, 2023.

[10] G. Qian, J. Mai, A. Hamdi, J. Ren, A. Siarohin, B. Li, H.-Y. Lee, I. Skorokhodov, P. Wonka, S. Tulyakov et al., âMagic123: One image to high-quality 3d object generation using both 2d and 3d diffusion priors,â in The Twelfth International Conference on Learning Representations, 2024.

[11] L. Melas-Kyriazi, I. Laina, C. Rupprecht, and A. Vedaldi, âRealfusion: 360deg reconstruction of any object from a single image,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 8446â8455.

[12] A. Haque, M. Tancik, A. A. Efros, A. Holynski, and A. Kanazawa, âInstruct-nerf2nerf: Editing 3d scenes with instructions,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 19 740â19 750.

[13] Y. Chen, Z. Chen, C. Zhang, F. Wang, X. Yang, Y. Wang, Z. Cai, L. Yang, H. Liu, and G. Lin, âGaussianeditor: Swift and controllable 3d editing with gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 476â21 485.

[14] X.-Y. Yu, J.-X. Yu, L.-B. Zhou, Y. Wei, and L.-L. Ou, âInstantstylegaussian: Efficient art style transfer with 3d gaussian splatting,â arXiv preprint arXiv:2408.04249, 2024.

[15] T. Brooks, A. Holynski, and A. A. Efros, âInstructpix2pix: Learning to follow image editing instructions,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 18 392â18 402.

[16] H.-P. Huang, H.-Y. Tseng, S. Saini, M. Singh, and M.-H. Yang, âLearning to stylize novel views,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 13 869â13 878.

[17] F. Mu, J. Wang, Y. Wu, and Y. Li, â3d photo stylization: Learning to generate stylized novel views from a single image,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16 273â16 282.

[18] K. Yin, J. Gao, M. Shugrina, S. Khamis, and S. Fidler, â3dstylenet: Creating 3d shapes with geometric and texture style variations,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 12 456â12 465.

[19] O. Michel, R. Bar-On, R. Liu, S. Benaim, and R. Hanocka, âText2mesh: Text-driven neural stylization for meshes,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 13 492â13 502.

[20] K. Zhang, N. Kolkin, S. Bi, F. Luan, Z. Xu, E. Shechtman, and N. Snavely, âArf: Artistic radiance fields,â in European Conference on Computer Vision. Springer, 2022, pp. 717â733.

[21] K. Liu, F. Zhan, Y. Chen, J. Zhang, Y. Yu, A. El Saddik, S. Lu, and E. P. Xing, âStylerf: Zero-shot 3d style transfer of neural radiance fields,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 8338â8348.

[22] K. Liu, F. Zhan, M. Xu, C. Theobalt, L. Shao, and S. Lu, âStylegaussian: Instant 3d style transfer with gaussian splatting,â in SIGGRAPH Asia 2024 Technical Communications, 2024, pp. 1â4.

[23] A. S. Kov Â´ acs, P. Hermosilla, and R. G. Raidou, âG-style: Stylized Â´ gaussian splatting,â in Computer Graphics Forum, vol. 43, no. 7. Wiley Online Library, 2024, p. e15259.

[24] M. Ye, M. Danelljan, F. Yu, and L. Ke, âGaussian grouping: Segment and edit anything in 3d scenes,â in European Conference on Computer Vision. Springer, 2024, pp. 162â179.

[25] O. Ronneberger, P. Fischer, and T. Brox, âU-net: Convolutional networks for biomedical image segmentation,â in Medical image computing and computer-assisted interventionâMICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer, 2015, pp. 234â241.

[26] J. J. Virtusio, J. J. M. Ople, D. S. Tan, M. Tanveer, N. Kumar, and K.-L. Hua, âNeural style palette: A multimodal and interactive style transfer from a single style image,â IEEE Transactions on Multimedia, vol. 23, pp. 2245â2258, 2021.

[27] S. Liu and T. Zhu, âStructure-guided arbitrary style transfer for artistic image and video,â IEEE Transactions on Multimedia, vol. 24, pp. 1299â 1312, 2022.

[28] H. Mun, G.-J. Yoon, J. Song, and S. M. Yoon, âTexture preserving photo style transfer network,â IEEE Transactions on Multimedia, vol. 24, pp. 3823â3834, 2022.

[29] Y. Huang, M. Jing, J. Zhou, Y. Liu, and Y. Fan, âLccstyle: Arbitrary style transfer with low computational complexity,â IEEE Transactions on Multimedia, vol. 25, pp. 501â514, 2023.

[30] H. Ding, H. Zhang, G. Fu, C. Jiang, F. Luo, C. Xiao, and M. Xu, âTowards high-quality photorealistic image style transfer,â IEEE Transactions on Multimedia, vol. 26, pp. 9892â9905, 2024.

[31] L. A. Gatys, A. S. Ecker, and M. Bethge, âImage style transfer using convolutional neural networks,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 2414â2423.

[32] X. Huang and S. Belongie, âArbitrary style transfer in real-time with adaptive instance normalization,â in Proceedings of the IEEE international conference on computer vision, 2017, pp. 1501â1510.

[33] Y. Jing, Y. Yang, Z. Feng, J. Ye, Y. Yu, and M. Song, âNeural style transfer: A review,â IEEE transactions on visualization and computer graphics, vol. 26, no. 11, pp. 3365â3385, 2019.

[34] K. Sohn, N. Ruiz, K. Lee, D. C. Chin, I. Blok, H. Chang, J. Barber, L. Jiang, G. Entis, Y. Li et al., âStyledrop: text-to-image generation in any style,â in Proceedings of the 37th International Conference on Neural Information Processing Systems, 2023, pp. 66 860â66 889.

[35] Z. Wang, X. Wang, L. Xie, Z. Qi, Y. Shan, W. Wang, and P. Luo, âStyleadapter: A unified stylized image generation model,â International Journal of Computer Vision, vol. 133, no. 4, pp. 1894â1911, 2025.

[36] A. Hertz, A. Voynov, S. Fruchter, and D. Cohen-Or, âStyle aligned image generation via shared attention,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 4775â4785.

[37] J. Chung, S. Hyun, and J.-P. Heo, âStyle injection in diffusion: A training-free approach for adapting large-scale diffusion models for style transfer,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 8795â8805.

[38] H. Wang, M. Spinelli, Q. Wang, X. Bai, Z. Qin, and A. Chen, âInstantstyle: Free lunch towards style-preserving in text-to-image generation,â arXiv preprint arXiv:2404.02733, 2024.

[39] J. Dong and Y.-X. Wang, âVica-nerf: View-consistency-aware 3d editing of neural radiance fields,â Advances in Neural Information Processing Systems, vol. 36, 2024.

[40] J. Zhuang, C. Wang, L. Lin, L. Liu, and G. Li, âDreameditor: Textdriven 3d scene editing with neural fields,â in SIGGRAPH Asia 2023 Conference Papers, 2023, pp. 1â10.

[41] J. Wu, J.-W. Bian, X. Li, G. Wang, I. Reid, P. Torr, and V. A. Prisacariu, âGaussctrl: Multi-view consistent text-driven 3d gaussian splatting editing,â in European Conference on Computer Vision. Springer, 2024, pp. 55â71.

[42] W. Liang, H. Xu, W. Gan, and W. Kang, âZero-shot text-driven dynamic neural radiance fields stylization,â IEEE Transactions on Multimedia, pp. 1â14, 2025.

[43] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., âLearning transferable visual models from natural language supervision,â in International conference on machine learning. PmLR, 2021, pp. 8748â8763.

[44] H.-W. Pang, B.-S. Hua, and S.-K. Yeung, âLocally stylized neural radiance fields,â in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE Computer Society, 2023, pp. 307â316.

[45] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., âSegment anything,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4015â4026.

[46] H. K. Cheng, S. W. Oh, B. Price, A. Schwing, and J.-Y. Lee, âTracking anything with decoupled video segmentation,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 1316â1326.

[47] J. L. Ba, J. R. Kiros, and G. E. Hinton, âLayer normalization,â arXiv preprint arXiv:1607.06450, 2016.

[48] L. Zhang, A. Rao, and M. Agrawala, âAdding conditional control to text-to-image diffusion models,â in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 3836â3847.

[49] J. Song, C. Meng, and S. Ermon, âDenoising diffusion implicit models,â in International Conference on Learning Representations, 2021.

[50] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Å. Kaiser, and I. Polosukhin, âAttention is all you need,â Advances in neural information processing systems, vol. 30, 2017.

[51] B. Mildenhall, P. P. Srinivasan, R. Ortiz-Cayon, N. K. Kalantari, R. Ramamoorthi, R. Ng, and A. Kar, âLocal light field fusion: Practical view synthesis with prescriptive sampling guidelines,â ACM Transactions on Graphics (ToG), vol. 38, no. 4, pp. 1â14, 2019.

[52] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, âTanks and temples: Benchmarking large-scale scene reconstruction,â ACM Transactions on Graphics (ToG), vol. 36, no. 4, pp. 1â13, 2017.

[53] Z. Teed and J. Deng, âRaft: Recurrent all-pairs field transforms for optical flow,â in Computer VisionâECCV 2020: 16th European Conference, Glasgow, UK, August 23â28, 2020, Proceedings, Part II 16. Springer, 2020, pp. 402â419.

[54] S. Niklaus and F. Liu, âSoftmax splatting for video frame interpolation,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 5437â5446.

[55] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 586â595.

[56] M. Ku, C. Wei, W. Ren, H. Yang, and W. Chen, âAnyv2v: A tuningfree framework for any video-to-video editing tasks,â Transactions on Machine Learning Research, 2024.

[57] Q. Song, M. Lin, W. Zhan, S. Yan, L. Cao, and R. Ji, âUnivst: A unified framework for training-free localized video style transfer,â arXiv preprint arXiv:2410.20084, 2024.