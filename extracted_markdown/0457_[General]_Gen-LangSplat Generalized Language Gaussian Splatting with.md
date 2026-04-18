# Gen-LangSplat: Generalized Language Gaussian Splatting with Pre-Trained Feature Compression

Pranav Saxena1

<!-- image-->  
Figure 1. Visualization of learned 3D language features of the previous SOTA method, LangSplat, and our proposed approach. Without requiring per-scene training for the language-feature autoencoder, our method achieves comparable, or even superior, results while being more efficient.

## Abstract

Modeling open-vocabulary language fields in 3D is essential for intuitive human-AI interaction and querying within physical environments. State-of-the-art approaches, such as LangSplat, leverage 3D Gaussian Splatting to efficiently construct these language fields, encoding features distilled from high-dimensional models like CLIP. However, this efficiency is currently offset by the requirement to train a scene-specific language autoencoder for feature compression, introducing a costly, per-scene optimization bottleneck that hinders deployment scalability. In this work, we introduce Gen-LangSplat, that eliminates this requirement by replacing the scene-wise autoencoder with a generalized autoencoder, pre-trained extensively on the largescale ScanNet dataset. This architectural shift enables the use of a fixed, compact latent space for language features across any new scene without any scene-specific training.

By removing this dependency, our entire language field construction process achieves a efficiency boost while delivering querying performance comparable to, or exceeding, the original LangSplat method. To validate our design choice, we perform a thorough ablation study empirically determining the optimal latent embedding dimension and quantifying representational fidelity using Mean Squared Error and cosine similarity between the original and reprojected 512-dimensional CLIP embeddings. Our results demonstrate that generalized embeddings can efficiently and accurately support open-vocabulary querying in novel 3D scenes, paving the way for scalable, real-time interactive 3D AI applications. The code can be found at https://github.com/Pranav-Saxena/Gen-LangSplat.

## 1. Introduction

Understanding and interacting with the 3D world through natural language is an emerging challenge at the intersection of 3D vision and multimodal learning. The ability to associate linguistic concepts with spatially grounded representations is essential for a range of applications in embodied AI, including open-vocabulary scene understanding, 3D semantic reasoning, and language-guided robotic manipulation. Earlier methods have primarily relied on Neural Radiance Fields (NeRFs) that distill vision-language model (VLM) features, such as those from CLIP [1], into volumetric neural fields. While these approaches enable openended language queries within reconstructed 3D scenes, the implicit nature of NeRFs introduces substantial inefficiencies, rendering is slow, optimization requires dense sampling, and the models are difficult to adapt or edit. These limitations have motivated the exploration of more explicit and computationally efficient 3D representations.

Recent advances in 3D Gaussian Splatting (3DGS) [2] have demonstrated remarkable efficiency and visual fidelity by representing scenes as a collection of spatially distributed anisotropic Gaussians. Building on this representation, LangSplat [3] introduced the concept of a 3D language field, where CLIP features are attached to individual Gaussians to support open-vocabulary querying in 3D. By replacing volumetric NeRF rendering with tile-based Gaussian rasterization, LangSplat achieves over two orders of magnitude faster performance while maintaining highquality alignment between image and language spaces. However, a key limitation of LangSplat lies in its reliance on per-scene autoencoders, which are trained separately for every scene to compress high-dimensional CLIP features into a low-dimensional latent space. This per-scene feature compression not only adds significant computational overhead but also prevents generalization to unseen scenes, since each environment requires new optimization to adapt the language embeddings.

In this work, we revisit this design bottleneck and propose a framework that eliminates the need for scene-specific autoencoders through the introduction of a generalized autoencoder trained across diverse scenes. Our method leverages a large-scale dataset, ScanNet [4], to train a unified encoderâdecoder network capable of learning a compact and transferable latent space for CLIP embeddings. Once trained, the generalized autoencoder can directly process CLIP features from new scenes without any fine-tuning, enabling rapid deployment across diverse environments. Integrating this generalized feature compression into a 3D Gaussian Splatting pipeline preserves the open-vocabulary grounding ability of LangSplat while improving overall efficiency by approximately 2Ã. Each Gaussian in our model jointly encodes appearance, geometry, and low-dimensional language features derived from the generalized embedding, which are decoded back to CLIP space during rendering to ensure multi-view consistency. This design not only accelerates training and inference but also establishes a consistent latent manifold shared across different scenes, facilitating scalable and cross-scene language reasoning in 3D.

To evaluate the representational strength of our generalized embedding, we perform a comprehensive ablation study that investigates both efficiency and reconstruction fidelity. We vary the latent dimensionality to analyze its effect on rendering quality, memory cost, and retrieval accuracy, and further quantify how closely reprojected CLIP embeddings approximate their original 512-dimensional counterparts using mean squared error (MSE) and cosine similarity metrics. Our findings reveal that a 16-dimensional embedding achieves the optimal balance between compactness and fidelity, outperforming the 3-dimensional latent space used in LangSplat in both reconstruction accuracy and semantic consistency. Despite removing the perscene training step, our method achieves comparable localization precision and open-vocabulary retrieval accuracy to LangSplat, confirming that a single generalized autoencoder is sufficient for language-grounded 3D reconstruction across diverse real-world environments.

In summary, our work establishes that cross-scene feature compression is a viable and efficient alternative to scene-specific optimization, making language-aware 3D Gaussian Splatting more practical and effective. By decoupling the language embedding process from per-scene adaptation, we demonstrate that robust open-vocabulary 3D understanding can be achieved at a fraction of the cost and time. We summarize the contributions of this paper as follows:

â¢ We introduce a cross-scene generalized autoencoder trained on the ScanNet dataset that learns a transferable latent representation for language features. Unlike prior per-scene approaches, our model eliminates the need for scene-specific retraining while maintaining strong semantic consistency across diverse environments.

â¢ Our method achieves nearly a 2Ã improvement in overall efficiency compared to LangSplat by removing the need to train a separate autoencoder for each scene.

â¢ We conduct a detailed ablation study on latent dimensionality to analyze its impact on feature reconstruction and semantic retention. The results show that a 16- dimensional latent embedding achieves high cosine similarity and low MSE when reprojected into the CLIP feature space, without compromising efficiency.

## 2. Related Work

## 2.1. 3D Scene Representations and Neural Rendering

Neural implicit representations have been widely adopted for high-quality 3D reconstruction and rendering. Early works such as DeepVoxels [5] and Neural Volumes [6] introduced volumetric neural rendering pipelines that model scene appearance as continuous functions. Neural Radiance

Fields (NeRFs) [7] further revolutionized this field by representing radiance and density using multilayer perceptrons, achieving photorealistic view synthesis from sparse inputs. However, the implicit nature of NeRFs incurs heavy computational cost due to dense ray sampling and global optimization, limiting their real-time applicability. Later extensions such as Mip-NeRF [8], Plenoxels [9], and Instant-NGP [10] improved speed and efficiency using mip-level sampling, voxel grids, and hash-based encodings, yet still rely on costly volume integration.

3D Gaussian Splatting (3DGS) [2] introduced a paradigm shift by representing scenes as explicit, spatially adaptive anisotropic Gaussians optimized through differentiable rasterization. Each Gaussian encapsulates geometry, color, opacity, and orientation, enabling real-time rendering with significantly reduced memory usage. Subsequent works have demonstrated the flexibility of this representation for applications in novel-view synthesis [11], dynamic scenes [12], and efficient reconstruction from sparse observations [13]. This explicit and differentiable formulation serves as the foundation for recent developments in semantically and linguistically grounded 3D understanding.

## 2.2. Language-Grounded 3D Representations

Integrating visionâlanguage understanding into 3D scene representations has become increasingly relevant for openvocabulary reasoning and interactive scene understanding. The success of large-scale visionâlanguage models such as CLIP [1] has inspired a series of methods that incorporate language embeddings into neural rendering frameworks. Early works like LERF [14] and Semantic-NeRF [15] embedded CLIP features within NeRF fields to enable openended text queries and semantic localization in 3D. Despite their strong alignment between vision and language modalities, these methods inherit NeRFâs inefficiencies in rendering and optimization, constraining their scalability to large or complex scenes.

LangSplat [3] overcomes these limitations by introducing a 3D Gaussian-based language field that attaches CLIP features to individual Gaussians, achieving precise and efficient open-vocabulary 3D querying. Through hierarchical SAM-based supervision [16], LangSplat delineates finegrained object boundaries and reduces feature bleeding across regions. However, its reliance on per-scene autoencoders for compressing 512-dimensional CLIP embeddings into a compact latent space significantly restricts scalability. Each scene requires a new autoencoder to be trained, resulting in additional compute overhead and lack of cross-scene generalization.

Our approach eliminates this bottleneck by introducing a generalized language autoencoder trained on Scan-Net [4], which directly generalizes across unseen environments without retraining.

## 2.3. 3D Grouping and Semantic Decomposition

Recent works have explored enhancing interpretability and compositional reasoning within Gaussian-based frameworks. Gaussian Grouping (GG) [17] introduces learnable identity embeddings per Gaussian, enabling zero-shot instance segmentation and 3D object grouping. Similar methods such as EditSplat [18] and SAGS [19] leverage SAM [16] or DINO [20] features to guide 3D semantic decomposition, allowing downstream applications such as object editing, removal, and part-level manipulation. While these methods enhance geometric interpretability, they do not incorporate language grounding and thus cannot support open-vocabulary querying. Our framework complements these efforts by coupling semantic reasoning with efficient, language-aware 3D Gaussian representations.

## 3. Preliminaries

3D Gaussian Representation. A 3D scene can be represented explicitly as a set of anisotropic Gaussians distributed in space, where each Gaussian $G _ { i } ( x )$ is defined by a mean $\mu _ { i } \in { \mathbb { R } } ^ { 3 }$ and a covariance matrix $\Sigma _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ :

$$
G _ { i } ( x ) = \exp \left( - \frac { 1 } { 2 } ( x - \mu _ { i } ) ^ { \top } \Sigma _ { i } ^ { - 1 } ( x - \mu _ { i } ) \right) .\tag{1}
$$

Each Gaussian additionally stores color $\mathbf { c } _ { i } \in \mathbb { R } ^ { 3 }$ , opacity $\alpha _ { i } .$ and a scale-rotation decomposition of $\Sigma _ { i }$ to model spatial extent and orientation. The full scene is thus represented as $\mathcal { G } = \{ ( \mu _ { i } , \Sigma _ { i } , \mathbf { c } _ { i } , \alpha _ { i } ) \} _ { i = 1 } ^ { N }$ 1 for N Gaussians.

Differentiable Rendering. Rendering proceeds by projecting each 3D Gaussian into the image plane using a perspective projection $\pi ( \cdot )$ parameterized by camera intrinsics and extrinsics. The projected 2D Gaussian contributes to pixel v according to its screen-space extent and accumulated opacity. The rendered color at pixel v is obtained through frontto-back alpha compositing:

$$
C ( v ) = \sum _ { i \in \mathcal { N } ( v ) } \mathbf { c } _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{2}
$$

where $\mathcal { N } ( v )$ denotes the ordered set of Gaussians intersecting the camera ray corresponding to pixel $v , c _ { i }$ is the color of the i-th Gaussian, and $\alpha _ { i } = o _ { i } G _ { i } ^ { 2 D } ( v )$ . Here $o _ { i }$ is the opacity of the i-th Gaussian and $G _ { i } ^ { 2 D } ( \cdot )$ represents the function of the i-th Gaussian projected onto 2D.. This formulation provides a fully differentiable rasterization pipeline that allows gradients to from image-space loss functions to the 3D Gaussian parameters.

Language Feature Embedding. To enable semantic reasoning and open-vocabulary understanding, each Gaussian is augmented with an additional feature vector $\mathbf { f } _ { i } \in \mathbb { R } ^ { d }$ derived from a visionâlanguage model such as CLIP. These feature vectors encode language-aligned semantics corresponding to the local image region or object represented by the Gaussian. During rendering, the per-pixel language feature is accumulated analogously to color:

<!-- image-->  
Figure 2. Overview of the proposed Gen-LangSplat framework. We leverage SAM to extract hierarchical semantics from multi-view images to resolve point ambiguity. The resulting segmentation masks are processed by the CLIP image encoder to obtain 512-D embeddings. These embeddings are compressed into a 16-D latent space using a generalized autoencoder pre-trained on ScanNet. Our 3D language Gaussians learn language features directly on a shared latent space derived from the pre-trained autoencoder. During querying, the rendered latent embeddings are decoded through the frozen decoder to recover the corresponding CLIP-space features for semantic reasoning.

$$
F ( v ) = \sum _ { i \in \mathcal { N } ( v ) } \mathbf { f } _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) .\tag{3}
$$

The resulting feature map $F ( v )$ can then be used for openvocabulary querying or similarity-based retrieval in the language embedding space.

Feature Compression. Since high-dimensional embeddings (e.g., 512-D CLIP features) are expensive to store and optimize for every Gaussian, a learned encoderâdecoder network is employed to compress the language features into a low-dimensional latent space $\mathbf { z } _ { i } \in \mathbb { R } ^ { k }$ , where $k \ll 5 1 2$ The encoder $E ( \cdot )$ and decoder $D ( \cdot )$ satisfy

$$
\begin{array} { r } { { \bf z } _ { i } = E ( { \bf f } _ { i } ) , \qquad \hat { \bf f } _ { i } = D ( { \bf z } _ { i } ) , } \end{array}\tag{4}
$$

and are trained to minimize reconstruction loss between $\hat { \mathbf { f } } _ { i }$ and fi:

$$
\mathcal { L } _ { \mathrm { r e c } } = \Vert \hat { \mathbf { f } } _ { i } - \mathbf { f } _ { i } \Vert _ { 2 } ^ { 2 } + \lambda ( 1 - \cos ( \hat { \mathbf { f } } _ { i } , \mathbf { f } _ { i } ) ) .\tag{5}
$$

This compression preserves semantic consistency while significantly reducing memory and computation costs, enabling scalable, language-aware 3D Gaussian rendering.

## 4. Methodology

Our method builds upon the LangSplat framework and extends it with a generalized language feature compression module to enable scalable, memory-efficient openvocabulary 3D reasoning. The overall pipeline is composed of three main stages: (1) Preprocessing, where hierarchical semantics and CLIP features are prepared; (2) Generalized Autoencoder, where a pre-trained cross-scene encoder compresses CLIP embeddings into a 16-dimensional latent space; and (3) Training of Language-Embedded Gaussians, where the pre-trained RGB Gaussians are taken as initialization, and only the language feature channels are optimized to learn the 3D language field within the Gaussian Splatting framework.

## 4.1. Preprocessing

Given a set of posed RGB images $\{ I _ { t } \} _ { t = 1 } ^ { T }$ , we begin by extracting hierarchical semantic information to mitigate the point ambiguity problem, where a single 3D point contributes to multiple overlapping semantic levels. To this end, we leverage the Segment Anything Model (SAM) to produce multi-scale hierarchical segmentation masks for each image. These masks serve as spatially coherent supervision regions that guide the extraction of semantically meaningful features.

Each segmented region is passed through the CLIP image encoder, resulting in a 512-dimensional embedding $\mathbf { f } _ { m } ~ \in ~ \mathbb { R } ^ { 5 1 2 }$ for each mask m. These embeddings capture the high-level semantic correspondence between visual regions and textual concepts. Compared to patch-based multi-scale methods that suffer from feature smoothing and boundary leakage, the use of SAM-based masks ensures sharper object boundaries and semantically aligned regions across multiple levels of granularity. The resulting set of CLIP embeddings forms the high-dimensional input feature space that will later be compressed using our generalized autoencoder.

## 4.2. Generalized Autoencoder

Instead of training a per-scene language autoencoder as in LangSplat, we propose a single generalized autoencoder trained across diverse indoor environments to learn a unified and transferable latent representation for CLIP features. The network consists of an encoder $E _ { \theta }$ that maps highdimensional CLIP embeddings to a compact latent space, and a decoder $D _ { \phi }$ that reconstructs them back to the original CLIP feature space:

$$
\begin{array} { r } { { \bf z } = E _ { \theta } ( { \bf f } ) , \quad \hat { \bf f } = D _ { \phi } ( { \bf z } ) , } \end{array}\tag{6}
$$

where $\mathbf { f } \in \mathbb { R } ^ { 5 1 2 }$ denotes the CLIP feature and $\mathbf { z } \in \mathbb { R } ^ { k }$ is the latent representation. Following LangSplat, the network is optimized using an $\ell _ { 1 }$ reconstruction loss combined with a cosine similarity term:

$$
\mathcal { L } _ { A E } = \| \mathbf { f } - \hat { \mathbf { f } } \| _ { 1 } + \lambda ( 1 - \cos ( \mathbf { f } , \hat { \mathbf { f } } ) ) ,\tag{7}
$$

ensuring both feature reconstruction accuracy and angular consistency within CLIP space.

The autoencoder is pre-trained on CLIP embeddings extracted from SAM-derived segmentation masks over Scan-Net scenes. For each image, SAM (ViT-H) generates dense mask proposals, and OpenCLIP ViT-B/16 encodes these masked regions into 512-dimensional features. Training across millions of such mask-level embeddings enables the autoencoder to capture category and scene-invariant semantic structure. This cross-scene supervision allows direct feature compression at test time without requiring any scenespecific optimization.

Through ablation, we find that a latent dimensionality of $k ~ = ~ 1 6$ retains over 93% cosine similarity to the original CLIP embeddings, while providing a significant reduction in storage and computational cost. After pretraining, both the encoder and decoder weights are frozen for all downstream tasks, allowing immediate deployment on novel scenes with no additional fine-tuning.

## 4.3. Training of Language-Embedded Gaussians

We integrate the generalized autoencoder into the 3D Gaussian Splatting framework to construct a compact 3D language field. Each Gaussian $G _ { i } = ( \mu _ { i } , \Sigma _ { i } , \mathbf { c } _ { i } , \alpha _ { i } , \mathbf { z } _ { i } )$ represents spatial position $\mu _ { i }$ , covariance $\Sigma _ { i } ,$ color coefficients $\mathbf { c } _ { i } ,$ opacity $\alpha _ { i } ,$ and a learnable latent language feature $\mathbf { z } _ { i }$ encoded by $E _ { \theta }$

Following the standard 3DGS pipeline, we first optimize all Gaussians for photometric reconstruction using RGB supervision. Once the RGB model converges, all geometric and appearance parameters $( \mu _ { i } , \Sigma _ { i } , \mathbf { c } _ { i } , \alpha _ { i } )$ are frozen, and only the latent language features $\mathbf { z } _ { i }$ are optimized. For each camera view, the rendered latent feature map is obtained by alpha compositing the latent codes:

$$
Z ( v ) = \sum _ { i \in \mathcal { N } ( v ) } \mathbf { z } _ { i } \alpha _ { i } ( v ) \prod _ { j < i } ( 1 - \alpha _ { j } ( v ) ) ,\tag{8}
$$

where $\mathcal { N } ( v )$ denotes the ordered set of Gaussians along the ray corresponding to pixel $v ,$ and $\alpha _ { i } ( v )$ is the viewdependent opacity. The target features are computed by encoding per-mask CLIP features of the input image using the frozen encoder $E _ { \theta }$ . The rendered and target latent maps are supervised using a combined $\ell _ { 1 }$ and cosine similarity loss:

$$
\mathcal { L } _ { L a n g } = \Vert Z ( v ) - H ( v ) \Vert _ { 1 } + \gamma ( 1 - \cos ( Z ( v ) , H ( v ) ) ) .\tag{9}
$$

The overall training objective is then:

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { R G B } + \beta \mathcal { L } _ { L a n g } ,\tag{10}
$$

where $\mathcal { L } _ { R G B }$ enforces photometric consistency from the RGB stage, and $\beta$ balances semantic supervision.

Unlike prior methods that train a separate autoencoder per scene, our design enables the 3D Gaussian language field to operate directly on a shared latent manifold. This leads to a two-fold reduction in total training time, consistent feature distributions across scenes, and comparable open-vocabulary localization performance to scene-specific approaches. During querying, the rendered latent embeddings are decoded via $D _ { \phi }$ into CLIP space, allowing direct text-based retrieval and open-vocabulary interaction.

## 5. Experiments and Results

## 5.1. Evaluation

Datasets. Similar to LangSplat, we evaluate our approach on two datasets. (i) LERF dataset [14], which consists of complex in-the-wild scenes designed for 3D object localization tasks. We use the extended version of LERF provided by LangSplat, which includes annotated ground-truth masks for textual queries, enabling open-vocabulary 3D semantic segmentation evaluation. Following LERF [14], we report localization accuracy for the 3D object localization task and Intersection-over-Union (IoU) for 3D semantic segmentation. (ii) 3D-OVS dataset [21], which contains a diverse collection of long-tail indoor objects captured under varying poses and backgrounds. This dataset is designed for open-vocabulary 3D semantic segmentation, where the full list of object categories is available. Unlike other methods that use the entire list to generate predicted masks, we use only the queried category to produce the corresponding masks. The mean Intersection-over-Union (mIoU) metric is used for evaluation on this dataset.

Implementation Details. To extract the language features for each image, we employ the OpenCLIP ViT-B/16 model. For segmentation, we use the Segment Anything Model (SAM) with the ViT-H backbone to generate hierarchical 2D masks. For each scene, we first train a 3D Gaussian Splatting (3DGS) model to reconstruct the RGB scene following the default hyperparameter settings in [2]. The model is trained for 30,000 iterations, resulting in approximately 2.5 million Gaussians per scene. Then we train our 3D language Gaussians by fixing all other parameters of 3D Gaussians such as mean and opacity. Only the language features are learnable during this stage. We train the language features for 30,000 iterations. Our generalized autoencoder is implemented as an MLP that compresses 512- dimensional CLIP embeddings into 16-dimensional latent features. The autoencoder is trained on the ScanNet dataset using an NVIDIA RTX 4090 GPU (24 GB VRAM).

<!-- image-->

<!-- image-->  
âPaper Napkinâ

<!-- image-->  
Figure 3. Qualitative comparison of open-vocabulary 3D object localization and segmentation results on the LERF dataset. The red points indicate model predictions, black dashed boxes denote the ground-truth annotations, and the bottom row shows the corresponding binary segmentation masks.

## 5.2. Results on the LERF dataset

Quantitative Results. We first evaluate our method on the LERF dataset to compare its performance against existing approaches. The localization results are presented in Table 1. Our method achieves comparable, and in some cases superior, performance to LangSplat, attaining an overall localization accuracy of 84.4%. Table 2 reports the IoU scores for 3D semantic segmentation, where our approach performs on par with LangSplat. These results demonstrate the effectiveness of our generalized autoencoder design, confirming that a single cross-scene feature compressor can replace scene-specific training without degrading performance.

<table><tr><td>Test Scene</td><td>LERF [14]</td><td>LangSplat [3]</td><td>Ours</td></tr><tr><td>ramen</td><td>62.0</td><td>73.2</td><td>72.8</td></tr><tr><td>figurines</td><td>75.0</td><td>80.4</td><td>80.7</td></tr><tr><td>teatime</td><td>84.8</td><td>88.1</td><td>88.9</td></tr><tr><td>waldo_kitchen</td><td>72.7</td><td>95.5</td><td>95.1</td></tr><tr><td>overall</td><td>73.6</td><td>84.3</td><td>84.4</td></tr></table>

Table 1. Localization accuracy (%) comparisons on LERF dataset.

Visualization Results. To visualize the learned 3D language field, we project the learned language features onto their principal components using PCA, following previous work [22]. The visualization results are shown in Figure 1. We observe that the features learned by our method exhibit similar, and in some cases superior, structure compared to LangSplat, particularly in the Teatime scene. Furthermore, we present qualitative results for 3D object localization and semantic segmentation in Figure 3.

<table><tr><td>Test Scene</td><td>LERF [14]</td><td>LangSplat [3]</td><td>Ours</td></tr><tr><td>ramen</td><td>28.2</td><td>51.2</td><td>50.9</td></tr><tr><td>figurines</td><td>38.6</td><td>44.7</td><td>45.3</td></tr><tr><td>teatime</td><td>45.0</td><td>65.1</td><td>65.8</td></tr><tr><td>waldo_kitchen</td><td>37.9</td><td>44.5</td><td>44.3</td></tr><tr><td>overall</td><td>37.4</td><td>51.4</td><td>51.6</td></tr></table>

Table 2. Quantitative comparisons of 3D semantic segmentation on the LERF dataset. We report the average IoU scores (%).

## 5.3. Results on the 3D-OVS dataset

Quantitative Results. We compare our method with other 2D and 3D state-of-the-art approaches on the 3D-OVS dataset, as shown in Table 3. Our method achieves performance comparable to or exceeding existing approaches. Specifically, it performs on par with LangSplat while surpassing 2D-based methods such as ODISE [23] and OV-Seg [24], as well as 3D-based methods including LERF [14] and 3D-OVS [25]. Notably, for this dataset, we generate object masks using only the query category, while other methods (e.g., 3D-OVS) require the complete list of categories for prediction. Overall, our approach attains an mIoU of 93.3%, demonstrating that the proposed generalized autoencoder effectively retains semantic consistency and enables accurate open-vocabulary 3D segmentation without per-scene training.

<!-- image-->  
Figure 4. Qualitative comparisons of 3D Segmentation on the 3D OVS Dataset.

<table><tr><td>Method</td><td>bed</td><td>bench</td><td>room</td><td>sofa</td><td>lawn</td><td>overall</td></tr><tr><td>LSeg [26]</td><td>56.0</td><td>6.0</td><td>19.2</td><td>4.5</td><td>17.5</td><td>20.6</td></tr><tr><td>ODISE [23] OV-Seg [24]</td><td>52.6 79.8</td><td>24.1 88.9</td><td>52.5 71.4</td><td>48.3 66.1</td><td>39.8 81.2</td><td>43.5 77.5</td></tr><tr><td>FFD [22]</td><td>56.6</td><td>6.1</td><td>25.1</td><td>3.7</td><td>42.9</td><td>26.9</td></tr><tr><td>LERF [14]</td><td>73.5</td><td>53.2</td><td>46.6</td><td>27</td><td>73.7</td><td>54.8</td></tr><tr><td>3D-OVS [21]</td><td>89.5</td><td>89.3</td><td>92.8</td><td>74</td><td>88.2</td><td>86.8</td></tr><tr><td>LangSplat [3]</td><td>92.5</td><td>94.2</td><td>94.1</td><td>90.0</td><td>96.1</td><td>93.4</td></tr></table>

Table 3. Quantitative comparisons of 3D semantic segmentation on the 3D-OVS dataset. We report the mIoU scores (%).

Open Vocabulary Segmentation. We perform open vocabulary segmentation on the 3D-OVS dataset given a language query. We present the results in Figure 4. Cosine similarity is used to evaluate correspondence between the rendered 3D language embeddings and the query text, used to retrieve gaussians with the highest semantic alignment.

## 5.4. Ablation Study

To analyze how effectively different latent dimensions retain semantic information from CLIP embeddings, we conduct an ablation study by varying the latent space dimension d . For each configuration, we evaluate the reconstructed

CLIP features obtained from the decoder by computing the mean squared error (MSE) and cosine similarity with respect to the original 512-dimensional embeddings. The results are shown in Figure 5.

<!-- image-->

<!-- image-->  
Figure 5. Ablation study on the latent dimensionality of our generalized autoencoder. As the dimension increases, the reconstruction error (MSE) decreases while the cosine similarity improves and saturates beyond d=16 , indicating an optimal trade-off between compactness and semantic retention.

This study aims to determine which latent dimension can best preserve the original CLIP feature information after compression and reprojection. As observed, increasing d consistently improves reconstruction quality, MSE decreases while cosine similarity increases and begins to saturate beyond $d = 1 6$ . At this point, the MSE reaches approximately $3 \times 1 0 ^ { - 4 }$ and cosine similarity exceeds 0.93, indicating that most semantic information from the original features is retained.

These findings suggest that a 16-dimensional latent space provides the best trade-off between representational fidelity and compactness, effectively preserving language semantics while maintaining computational and memory efficiency.

## 6. Conclusion

In this paper, we presented Gen-LangSplat, a generalized framework for efficient, language-grounded 3D Gaussian Splatting. By replacing scene-specific autoencoders with a single pre-trained feature compression module, our method removes the need for per-scene training while maintaining comparable or superior performance to the SOTA, LangSplat, achieving nearly 2Ã higher overall efficiency and reduced memory cost. Furthermore, extensive ablation studies show that a 16-dimensional latent embedding provides the best trade-off between semantic retention and efficiency, enabling scalable and generalizable openvocabulary 3D understanding.

## References

[1] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748â8763. PMLR, 2021. 2, 3

[2] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. TOG, 42(4):1â14, 2023. 2, 3, 5

[3] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d language gaussian splatting. arXiv preprint arXiv:2312.16084, 2023. 2, 3, 6, 7

[4] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias NieÃner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proc. Computer Vision and Pattern Recognition (CVPR), IEEE, 2017. 2, 3

[5] Vincent Sitzmann, Justus Thies, Felix Heide, Matthias NieÃner, Gordon Wetzstein, and Michael Zollhofer. Deep- Â¨ voxels: Learning persistent 3d feature embeddings. In Proc. CVPR, 2019. 2

[6] Stephen Lombardi, Tomas Simon, Jason Saragih, Gabriel Schwartz, Andreas Lehrmann, and Yaser Sheikh. Neural volumes: Learning dynamic renderable volumes from images. ACM Trans. Graph., 38(4):65:1â65:14, July 2019. 2

[7] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 3

[8] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. ICCV, 2021. 3

[9] Sara Fridovich-Keil and Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In CVPR, 2022. 3

[10] Thomas Muller, Alex Evans, Christoph Schied, and Alexan- Â¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. TOG, 41(4):1â15, 2022. 3

[11] Kai Cheng, Xiaoxiao Long, Kaizhi Yang, Yao Yao, Wei Yin, Yuexin Ma, Wenping Wang, and Xuejin Chen. Gaussianpro: 3d gaussian splatting with progressive propagation. In Fortyfirst International Conference on Machine Learning. 3

[12] Jiawei Yang, Marco Pavone, and Yue Wang. 3

[13] Haolin Xiong, Sairisheek Muttukuru, Rishi Upadhyay, Pradyumna Chari, and Achuta Kadambi. Sparsegs: Realtime 360Â° sparse view synthesis using gaussian splatting. Arxiv, 2023. 3

[14] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. Lerf: Language embedded radiance fields. In ICCV, pages 19729â19739, 2023. 3, 5, 6, 7

[15] Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, and Andrew J Davison. In-place scene labelling and understanding with implicit scene representation. In ICCV, pages 15838â 15847, 2021. 3

[16] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, 2023. 3

[17] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke. Gaussian grouping: Segment and edit anything in 3d scenes. In ECCV, 2024. 3

[18] Dong In Lee, Hyeongcheol Park, Jiyoung Seo, Eunbyung Park, Hyunje Park, Ha Dam Baek, Shin Sangheon, Sangmin Kim, and Sangpil Kim. Editsplat: Multi-view fusion and attention-guided optimization for view-consistent 3d scene editing with 3d gaussian splatting. arXiv preprint arXiv:2412.11520, 2024. 3

[19] Xu Hu, Yuxi Wang, Lue Fan, Junsong Fan, Junran Peng, Zhen Lei, Qing Li, and Zhaoxiang Zhang. Semantic anything in 3d gaussians. arXiv preprint arXiv:2401.17857, 2024. 3

[20] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J Â´ egou, Â´ Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, pages 9650â9660, 2021. 3

[21] Kunhao Liu, Fangneng Zhan, Jiahui Zhang, Muyu Xu, Yingchen Yu, Abdulmotaleb El Saddik, Christian Theobalt, Eric Xing, and Shijian Lu. Weakly supervised 3d openvocabulary segmentation. In NeurIPS, 2023. 5, 7

[22] Sosuke Kobayashi, Eiichi Matsumoto, and Vincent Sitzmann. Decomposing nerf for editing via feature field distillation. NeurIPS, 35:23311â23330, 2022. 6, 7

[23] Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models. arXiv preprint arXiv:2303.04803, 2023. 6, 7

[24] Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan Zhao, Hang Zhang, Peizhao Zhang, Peter Vajda, and Diana Marculescu. Open-vocabulary semantic segmentation with mask-adapted clip. In CVPR, pages 7061â7070, 2023. 6, 7

[25] Youquan Liu, Lingdong Kong, Jun Cen, Runnan Chen, Wenwei Zhang, Liang Pan, Kai Chen, and Ziwei Liu. Segment any point cloud sequences by distilling vision foundation models. arXiv preprint arXiv:2306.09347, 2023. 6

[26] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and Rene Ranftl. Language-driven semantic segmentation. In ICLR, 2022. 7