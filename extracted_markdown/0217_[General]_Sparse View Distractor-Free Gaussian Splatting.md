# Sparse View Distractor-Free Gaussian Splatting

Yi Gu1,â, Zhaorui Wang1,â, Jiahang Cao1, Jiaxu Wang1, Mingle Zhao2, Dongjun Ye1, Renjing Xu1â 

Abstractâ 3D Gaussian Splatting (3DGS) enables efficient training and fast novel view synthesis in static environments. To address challenges posed by transient objects, distractorfree 3DGS methods have emerged and shown promising results when dense image captures are available. However, their performance degrades significantly under sparse input conditions. This limitation primarily stems from the reliance on the color residual heuristics to guide the training, which becomes unreliable with limited observations. In this work, we propose a framework to enhance distractor-free 3DGS under sparse-view conditions by incorporating rich prior information. Specifically, we first adopt the geometry foundation model VGGT to estimate camera parameters and generate a dense set of initial 3D points. Then, we harness the attention maps from VGGT for efficient and accurate semantic entity matching. Additionally, we utilize Vision-Language Models (VLMs) to further identify and preserve the large static regions in the scene. We also demonstrate how these priors can be seamlessly integrated into existing distractor-free 3DGS methods. Extensive experiments confirm the effectiveness and robustness of our approach in mitigating transient distractors for sparse-view 3DGS training.

## I. INTRODUCTION

Photorealistic 3D reconstruction from multi-view images are fundamental problem in computer vision and computer graphics, with a broad range of robotics-related applications, e.g., virtual reality [1], motion planning [2], [3], [4], navigation [5], [6], etc. Neural Radiance Fields (NeRF) [7] have achieved significant breakthroughs by leveraging implicit representations and volume rendering [8] to improve view synthesis quality. 3D Gaussian Splatting (3DGS) [9] introduces an explicit point-based representation that supports fast training and enables high-quality, real-time rendering. Despite these advances, 3DGS is primarily designed for static scenes and typically relies on densely captured images to reconstruct detailed 3D geometry.

Existing studies have explored sparse-view reconstruction by incorporating co-visibility and geometry constraints [10], [11] or leveraging data-driven feedforward approaches [12], [13], [14], primarily targeting static scenes. Another line of research aims to extend NeRF or 3DGS to dynamic environments where transient objects are present. Most distractor-free methods rely on explicit or implicit color residual heuristics [15], [16] to guide the training process. However, such heuristics tend to be less effective under sparse-view conditions, raising an important question: Can transient information be identified before the optimization?

In this work, we investigate how to extract rich mask prior information to enhance sparse-view 3D reconstruction. Given the superiority of 3DGS, we adopt it as our primary representation and demonstrate how our method facilitates distractor-free 3DGS training. To train a 3DGS model, the conventional data preparation pipeline typically relies on Structure from Motion (SfM) tools such as COLMAP [17] to estimate accurate camera parameters and generate sparse points. However, this approach often fails in the sparse-view setting or scenes with significant transient contents. Even when image registration succeeds, the resulting point cloud is usually too sparse to support high-quality reconstruction.

To improve the robustness of data preparation, we leverage the geometry foundation model VGGT [18] to estimate camera parameters. VGGT not only runs significantly faster but also produces a dense initial point cloud. However, this initial point cloud often includes many transient objects, complicating downstream processing. Interestingly, we observe that the attention maps in VGGT are highly informative and capable of consistently identifying the same objects across views without manual supervision. We exploit this property to automatically match corresponding objects and use the geometric outputs from VGGT to further validate the matching pairs. As a result, we can assign each image a prior static mask, where the unmasked regions correspond to potentially transient contents.

To improve the accuracy of prior masks, we investigate the potential of powerful Vision-Language Models (VLMs) to assist in our task. We encode our objectives into text prompts and directly input them alongside the images into VLMs. However, we observe that modern VLMs often misinterpret the task and may hallucinate irrelevant regions. To mitigate these issues, we simplify the problem by restricting VLM queries to large unmasked areas only. Experimental results show that our method generates highly accurate prior masks, significantly outperforming baseline approaches.

To demonstrate how these priors can be seamlessly integrated into existing distractor-free 3DGS framework, we incorporate them into RobustGS [19]. We introduce a simple yet effective warm-up strategy guided by mask priors, where the initial training masks in GS are replaced with our prior masks. Experimental results show that this approach yields substantial improvements, including a 1â4 dB gain in PSNR and significantly enhanced handling of distractors.

Our key contributions are as follows:

â¢ We propose an efficient and robust method for generating static masks by leveraging attention maps from VGGT to match corresponding mask pairs.

â¢ We explore the use of Vision-Language Models for the distractor-free reconstruction and introduce a reliable prompting strategy to generate high-quality prior masks.

â¢ We demonstrate how these prior masks can be seamlessly integrated into existing distractor-free 3DGS frameworks and achieve state-of-the-art results.

## II. RELATED WORK

Distractors handling in NeRF and 3DGS. The original NeRF [7] relies on strong assumptions about the capture setup, requiring the scene to remain perfectly static and the lighting conditions to stay consistent throughout the capture. Two pioneer works, NeRF-W [20] and RobustNeRF [15], extend NeRF to unstructured âin-the-wildâ captured images by using photometric error as guidance. RobustNeRF approached the problem from a robust estimator perspective, with binary weights determined by thresholded rendering error, and a blur kernel to reflect the assumption that pixels belonging to distractors are spatially correlated. However, both RobustNeRF and NeRF-W variants rely solely on RGB residual errors, which often leads to misclassification of transients that share similar colors with the background. NeRF-HuGS [16] combines heuristics from COLMAPâs sparse point cloud and off-the-shelf semantic segmentation to remove distractors, but both heuristics are shown to fail under heavy transient occlusions. NeRF On-the-go [21] uses semantic features from DINOv2 [22] to predict outlier masks via a lightweight MLP. However, it also relies on direct supervision from structural rendering errors, leading to potential over- or under-masking of outliers.

Following the evolution of Distractor-free NeRF methods, multiple works [23], [19] also use color residual heuristics as the transients guidance in the Gaussian Splatting. Benefiting from the explicit properties, some works [24], [25] propose to use view-specific Gaussian points for modeling per-view distractors, which utilize color residual heuristics implicitly. Extending the robustness to unconstrained photo collections, RobustSplat [26] and RobustSplat++ [27] introduce uncertainty modeling to mask out transient occluders alongside appearance variations. Similarly, DeGauss [28] proposes a decomposed framework that decouples dynamic entities from static scenes by leveraging semantic feature fields. Other works [29], [30] also incorporate DINOv2 [22] or Stable Diffusion [31] feature priors [23] to handle occlusions. In this work, we focus on unordered sparse images with minimal appearance changes. Thus, we exclude methods that primarily address appearance changes [32], [33] or rely on temporal information [34], [35], [36].

3DGS with geometry foundation models. While neural 3D reconstruction has made significant progress, it often relies on densely captured multi-view inputs with carefully initialized camera poses, typically obtained via Structurefrom-Motion (SfM) tools such as COLMAP [17]. This dependence limits its applicability in real-world scenarios, where sparse-view inputs and limited feature matches can lead to unreliable pose estimation and cumulative reconstruction errors. To address these challenges, InstantSplat [37] introduces a novel and extremely fast neural reconstruction pipeline built upon the Geometry Foundation models DUSt3R [38]/MASt3R [39], capable of recovering accurate 3D representations from as few as 2â3 input images. However, InstantSplat remains focused on static scenes. While concurrent SparseGS-W [40] leverages DUSt3R and diffusion models [31] for initialization and enhancement, it relies on manual, user-defined text prompts (e.g., âtouristâ) to generate mask priors. In contrast, our method is a more general and fully automated process that eliminates the need for pre-defined scene-specific prompts. A closely related effort is Easi3R [41], which also explores the role of attention mechanisms in DUSt3R. Unlike Easi3R, our method is independently developed and specifically targets unordered, sparsely captured images without assuming any available motion or temporal consistency.

## III. PRELIMINARIES

3DGS [9] explicitly represents 3D scenes with a set of 3D Gaussians {Gi}. Each Gaussian is defined by a Gaussian function:

$$
\begin{array} { r } { \mathscr { G } _ { i } ( { \pmb x } | { \pmb \mu } _ { i } , { \pmb \Sigma } _ { i } ) = e ^ { - \frac { 1 } { 2 } ( { \pmb x } - { \pmb \mu } _ { i } ) ^ { \top } { \pmb \Sigma } _ { i } ^ { - 1 } ( { \pmb x } - { \pmb \mu } _ { i } ) } , } \end{array}
$$

where $\mu _ { i } \in \mathbb { R } ^ { 3 }$ and $\pmb { \Sigma } _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ are the center of a point $p _ { i } \in$ $\mathcal { P }$ and corresponding 3D covariance matrix, respectively. The covariance matrix $\Sigma _ { i }$ can be decomposed into a scaling matrix $S _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ and a rotation matrix $\pmb { R } _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ :

$$
\pmb { \Sigma } _ { i } = \pmb { R } _ { i } \pmb { S } _ { i } \pmb { S } _ { i } ^ { \top } \pmb { R } _ { i } ^ { \top } .
$$

3DGS allows fast Î±-blending for rendering. Given a transformation matrix W and an intrinsic matrix K, $\mu _ { i }$ and $\Sigma _ { i }$ can be transformed to camera coordinate corresponding to W and then projected to 2D coordinates:

$$
\begin{array} { r } { \pmb { \mu } _ { i } ^ { ' } = \pmb { K } \pmb { W } [ \pmb { \mu } _ { i } , 1 ] ^ { \top } , \quad \pmb { \Sigma } _ { i } ^ { ' } = \pmb { J } \pmb { W } \pmb { \Sigma } _ { i } \pmb { W } ^ { \top } \pmb { J } ^ { \top } , } \end{array}
$$

where J denotes the Jacobian matrix of the projective transformation. Rendering color $C \in \mathbb { R } ^ { 3 }$ of a pixel u can be obtained in a manner of Î±-blending:

$$
\pmb { C } = \sum _ { i \in N _ { \mathscr { G } } } T _ { i } \alpha _ { i } \pmb { c } _ { i } , \quad T _ { i } = \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { i } ) ,
$$

where $\alpha _ { i }$ is calculated by evaluating $\mathcal { G } _ { i } ( \boldsymbol { \boldsymbol { u } } | \mu _ { i } ^ { ' } , \Sigma _ { i } ^ { ' } )$ multiplied with a learnable opacity corresponding to $\mathcal { G } _ { i }$ , and the viewdependent color $\boldsymbol { c } _ { i } \in \mathbb { R } ^ { 3 }$ is represented by spherical harmonics (SH) from the Gaussian $\mathcal { G } _ { i } , T _ { i }$ is the cumulative opacity. $N _ { \mathcal { G } }$ is the number of Gaussians that the ray passes through.

VGGT [18] is a feed-forward neural network that directly infers camera parameters, point maps, depth maps, and point tracks, from multiple views. Specifically, given N RGB images $( I _ { i } ) _ { i = 1 } ^ { N } , \ I _ { i } \ \in \ \mathbb { R } ^ { 3 \times H \times W }$ VGGTâs transformer is a function that maps this sequence to a corresponding set of 3D annotations, one per frame:

$$
f \left( ( I _ { i } ) _ { i = 1 } ^ { N } \right) = \left( \mathbf { g } _ { i } , D _ { i } , P _ { i } , T _ { i } \right) _ { i = 1 } ^ { N } .\tag{1}
$$

$\mathbf { g } _ { i } \in \mathbb { R } ^ { 9 }$ is the camera parameters of image $I _ { i } .$ The world reference frame is defined in the coordinate system of the first camera $\mathbf { g } _ { 1 }$ . The depth map $D _ { i } \in \mathbb { R } ^ { H \times W }$ , and point map $P _ { i } \in \mathbb { R } ^ { 3 \times H \times W }$ are redundancy designs to ease the training. Unless mentioned otherwise, we only use the depth map $D _ { i }$ to estimate dense geometry, which usually leads to more accurate 3D points than the point map branch.

The input image $I _ { i }$ of VGGT is initially patchified into a set of K tokens where $\mathrm { t } _ { i } ^ { I } \ \in \ \mathbb { R } ^ { K \times C }$ . The combined set of image tokens from all frames, i.e., $\mathrm { t } ^ { I } ~ = ~ \cup _ { i = 1 } ^ { N } \{ \mathrm { t } _ { i } ^ { I } \}$ , is subsequently processed through alternating frame-wise and global self-attention layers [42]. The frame-wise selfattention learns the monocular geometry by attending to the tokens $\mathrm { t } _ { k } ^ { I }$ within each frame separately. The global selfattention attends to the tokens $\mathrm { t } ^ { \bar { I } }$ across all frames jointly, which aggregates information from multiple views without any inductive bias.

## IV. METHOD

## A. Method Overview

Given N color images, we first employ VGGT to obtain the initial point cloud and camera parameters. Then, we use a class-agnostic mask predictor to process each image and derive the 2D masks $\{ m _ { i , k } | k = 1 , . . . , n _ { i } \}$ , where $n _ { i }$ denotes the number of masks in the image $I _ { i } .$ As presented in previous works [19], SAM [43] tends to over-segment the image. In this work, we use CropFormer [44] as our mask predictor, since the entity-level segmentation is well suited for the distractor-free task [45]. Considering the color residual heuristics are not reliable in the sparse-view setting, we decide to incorporate rich mask prior information before the GS training. Specifically, we utilize the attention map from VGGT to achieve fast entity matching (Sec. IV-B). After that, we resort to the VLM model to identify and preserve the large static regions in the scene (Sec. IV-C). We demonstrate how to apply our mask priors to existing distractor-free GS frameworks in Sec. IV-D.

<!-- image-->  
Fig. 1: Illustration of VGGT attention-guided semantic entity matching. Query tokens are highlighted in cyan. Initially, we project these query tokens onto the reference image to obtain the projected tokens. The reprojected tokens are computed in a similar manner. A projected token is considered valid only if its reprojected counterpart lies within the region of the query tokens; otherwise, it is classified as an invalid token (colored in red). To explicitly illustrate the cross-view correspondence matching process, we visualize the global feature maps. Compared to static objects, transient objects typically exhibit lower recall, which serves as a primary criterion for identifying distractors.

## B. VGGT-attention Guided Entity Matching

Our primary objective is to estimate the static mask priors prior to 3DGS training. Building on the assumption from NeRF-HuGS [16], the observation frequency of static objects should be higher than that of transient objects. However, relying on SfM-based heuristics, as proposed in NeRF-HuGS, may be less reliable in sparse-view settings due to the limited number of matching pairs and the presence of numerous outliers. In the context of VGGT-based initialization, geometry reprojection methods are constrained by the accuracy of VGGT. Alternatively, we can leverage the VGGT tracking head to achieve this goal. However, tracking too many points proves inefficient, and determining the confidence and visibility thresholds for VGGT tracking results remains challenging. Semantic-level matching, as proposed in SpotLessSplat [23], may be less effective in the presence of repetitive objects. Based on these observations, we aim to address this issue at both the geometric and semantic levels.

VGGT learns the high-level feature matching implicitly in the global self-attention layers. Given a pair of images, the query image $I _ { i }$ and the reference image $I _ { j }$ , where $j \neq i .$ We extract the query tokens corresponding to the mask $m _ { i , k }$ according to whether the patch is occupied by the mask. By setting the number of query tokens in each layer as $S ,$ the query tokens can be represented as $\mathrm { t } _ { i , k , l } ^ { I } \in \bar { \mathbb { R } ^ { S \times C _ { f } } }$ , where $C _ { f }$ is the feature dimension and l is the layer index. Then, we use $\mathrm { t } _ { i , k } ^ { I }$ to attend $\mathrm { t } _ { j } ^ { I } \mathrm { : }$

$$
\begin{array} { r } { \mathcal { A } = \mathrm { t } _ { i , k } ^ { I } \mathrm { t } _ { j } ^ { I ^ { T } } / \sqrt { C _ { f } } . } \end{array}\tag{2}
$$

Here we reshape A as $\mathcal { A } \in \mathbb { R } ^ { S \times L \times h \times w }$ , where $L$ is the number of global attention layers and h, w are the height and width of feature maps. By averaging the layer dimension, we will get an aggregated attention map $\bar { \mathcal { A } } \in \mathbb { R } ^ { S \times h \times w }$ . Then, we obtain the best matching tokens by selecting the index in $I _ { j }$ with the highest attention value:

$$
I n d e x _ { j , s } = \arg \operatorname* { m a x } _ { s \in \{ 1 , . . . , S \} } ( \bar { \mathcal { A } } ) .\tag{3}
$$

This operation can be viewed as a projection process, which considers the relevance at both feature and geometric levels. By index on Indexj,s, we can obtain the projected tokens $\mathrm { t } _ { j } ^ { \bar { p r o j } _ { i , k } }$

To filter the noised unrelated tokens in $\mathrm { t } _ { j } ^ { p r o j _ { i , k } }$ , we reproject the $\mathrm { t } _ { j } ^ { p r o j _ { i , k } }$ to attend the image $I _ { i } .$ . The re-projected tokens are represented as $\mathrm { t } _ { i } ^ { r e p _ { j } }$ . We only keep the tokens that have intersected regions with mask $\mathbf { m } _ { i , k }$ as the valid tokens, represented as $\tilde { \mathrm { t } } _ { i } ^ { p r o j _ { i , k } }$ and $\tilde { \ t } _ { i } ^ { r e p _ { j } }$ . We refer readers to Fig. 1 for a comprehensive illustration.

Our main observation is that the transient query mask tends to have lower recall, which can be calculated by:

$$
R e c a l l = | \tilde { \mathrm { t } } _ { i } ^ { r e p _ { j } } | / | \mathrm { t } _ { j } ^ { p r o j _ { i , k } } | ,\tag{4}
$$

where || denotes the number of tokens. By simply setting the recall threshold to 0.5, we can efficiently obtain the matching pair candidates. To further validate each pair, we use the Chamfer Distance (CD) to evaluate each pair. Using all points in the mask region is time-consuming, as the Chamfer Distance computation is relatively slow. Since the validated tokens have filtered out many irrelevant regions, we only compute the CD within the token masks.

<!-- image-->  
Fig. 2: VLM process illustraion. To simplify annotations, we exclude masks containing fewer than 20000 pixels. For the remaining transient candidate masks, we automatically assign a unique identifier to the center of each mask and highlight each mask with a random color. These operations, in combination with our prompts, significantly enhance the generation of mask priors.

We adopt a score-based method for entity matching, considering only pairs with a Chamfer Distance below the threshold $T h r e s h o l d _ { C D }$ across all datasets. Since a lower CD indicates a more confident match, we define a normalized matching score as:

$$
S c o r e = ( T h r e s h o l d _ { C D } - C D ) / T h r e s h o l d _ { C D } .\tag{5}
$$

After matching, each mask is assigned a confidence score. Masks with a score above $T h r e s h o l d _ { s c o r e }$ are selected as static masks, while the rest are treated as transient candidates.

## C. VLM Enhanced Mask Generation

On one hand, as discussed in the introduction and confirmed by our experiments, current VLMs exhibit some zero-shot capability in identifying transient objects, but their performance is unstableâparticularly when the number of transient objects is large. Therefore, directly combining VLMs with Grounding-SAM [46] may lead to suboptimal results. On the other hand, VGGT struggles to accurately predict geometry in large textureless regions such as sky, plain-colored curtains, or ground surfaces. Interestingly, we find that VLMs can easily interpret these regions. To leverage this complementary strength, we incorporate VLMs to enhance mask prior generation. Inspired by prior works [47], we automatically generate labeled identifiers in large mask regions and prompt the VLM to determine whether each labeled region is static or transient. We further query the VLM to provide a concise and accurate analysis, thereby activating its Chain-of-Thought capabilities [48], [49]. Fig. 2 illustrates the detailed VLM process.

## D. Integrated with RobustGS

RobustGS [19] uses logistic regression to flexibly learn the decision boundary thresholds. The input of the mask generation model is the residuals from the last iteration. Each iteration contains two back-propagation processes, one for the mask model and the other for GS parameters. RobustGS uses the masked image loss to optimize the mask generation model:

$$
\mathcal { L } _ { m a s k } = \hat { \mathcal { M } } \circ \mathcal { L } _ { G S } + \mathcal { L } _ { r e g } ,\tag{6}
$$

where ${ \mathcal { L } } _ { G S }$ contains L1 loss and SSIM loss and â¦ is the Hadamard product. $\mathcal { L } _ { r e g }$ is the regularization loss to avoid non-trivial solutions, such as classifying every pixel as a distractor. MË is the prediction of the mask model.

During the GS optimization, the mask $\hat { \mathcal { M } }$ is replaced with a non-learnable mask M by calculating the intersections with SAM [43] predictions. Thus, the loss function in the GS optimization stage is changed into:

$$
\mathcal { L } _ { m a s k } ^ { \prime } = \mathcal { M } \circ \mathcal { L } _ { G S } .\tag{7}
$$

Since the mask $\hat { \mathcal { M } }$ is derived from the color residual heuristics, most regions exhibit similar residual values during the initial training stage. This makes it difficult for the simple regression model to distinguish between transient and static regions. M, which is obtained from $\hat { \mathcal { M } } ,$ inherits this limitation.

To tackle this problem, we replace M with our mask priors during the early stage of GS training, serving as a warm-up phase for RobustGS. We do not introduce any additional loss terms, such as an L1 loss, to explicitly align MË with the transient mask priors, because the priors only serve as pseudo ground truth, and the simple regression model lacks sufficient representational power.

To address the inaccuracy of the initial camera parameters, we incorporate Gaussian Bundle Adjustment (GSBA) [37] during training to further refine the camera poses. Thus, the loss of ${ \mathcal { L } } _ { G S }$ is replaced by:

$$
\mathcal { L } _ { G S B A } = \mathcal { L } _ { G S } ( \{ \mathcal { G } _ { i } \} , \{ \mathcal { T } _ { j } \} ) ,\tag{8}
$$

where $\tau _ { j }$ is the extrinsic of the j-th camera. We empirically observe that allowing camera intrinsics to be optimized or sharing camera parameters across views leads to degraded performance, often resulting in increased ghosting artifacts. Therefore, we keep the intrinsics fixed throughout training. At test time, we perform a test-time optimization where only the camera poses are updated. The initial camera parameters for the test views are obtained by rerunning VGGT on both the training and testing images.

## V. IMPLEMENTATION, EVALUATION, AND RESULTS

Datasets. We select 5 scenes from the RobustNeRF [15] dataset and 6 scenes from the NeRF on-the-go dataset [21] to evaluate our method. Each scene consists of 6 experiments, where 4, 6, and 8 views are sampled twice. The view sampling process is implemented as a clustering procedure: the first frame is randomly selected, and subsequent views are added to the cluster only if they share more than 20 co-visible points with the already selected views.

<!-- image-->  
Fig. 3: Qualitative evaluation of baseline methods and our approach on the NeRF On-the-go and RobustNeRF datasets. \* means with the VGGT initialization.

Baselines and metrics. For mask priors, we compare our method with existing VLM models coupled with Grounding-SAM [46] and NeRF-HuGS [16]. The results for NeRF-HuGS are obtained by re-running COLMAP using only the training views. We use Intersection-of-Union (IoU) to evaluate each method. Due to only the Crab and Yoda scenes containing paired photos from identical camera poses, one with distractors present and another without and refine [19] it with the 2D mask lists $\{ \boldsymbol { m } _ { i , k } | k = 1 , . . . , n _ { i } \}$

For distractor-free rendering, we use WildGaussians [29], SpotLessSplat [23], RobustGS [19], and Desplat [25] as baseline methods. For comparison, we compute standard image reconstruction metrics (i.e., PSNR). All baseline methods are evaluated using their official implementations. Rerunning COLMAP on each sparse-view scenario leads to a high failure rate, and integrating each method with GSBA is a non-trivial task. Therefore, we use the camera parameters and initial points provided by the original densely captured datasets for all baselines. For clarity, an asterisk (\*) denotes our VGGT-based initialization in the following sections. We further add the comparison with RobustGS [19] with VGGT initialization for more fair comparison.

Implementation details. The $T h r e s h o l d _ { C D }$ is set as 0.2 across all datasets, and $T h r e s h o l d _ { s c o r e }$ is set as $0 . 5 \times N$ where N is the number of training views. For VLMs enhancement, we only query the regions larger than 20,000 pixels. All prompts are defined in a general manner, without any scene-specific customization [40]. We initialize the camera parameters and 3D points using VGGT, and filter the points with our high-quality mask priors. All experiments are trained for 10,000 iterations, with the first 500 iterations designated as the warm-up stage.

Corner  
Fountain  
Crab2  
<!-- image-->

<table><tr><td>Methods</td><td colspan="3">Yoda</td><td colspan="3">Crab1</td><td colspan="3">Crab2</td></tr><tr><td></td><td>4 views</td><td>6 views</td><td>8 views</td><td>4 views</td><td>6 views</td><td>8 views</td><td>4 views</td><td>6 views</td><td>8 views</td></tr><tr><td colspan="10">Closed-source VLM Models + Grounding SAM</td></tr><tr><td>GPT-4.1</td><td>43.75</td><td>45.81</td><td>54.49</td><td>12.06</td><td>18.23</td><td>9.99</td><td>11.41</td><td>14.46</td><td>0.00</td></tr><tr><td>GPT-40</td><td>32.91</td><td>20.15</td><td>50.70</td><td>10.35</td><td>2.66</td><td>15.33</td><td>1.38</td><td>0.00</td><td>0.00</td></tr><tr><td>Claude-3.7-Sonnet</td><td>59.55</td><td>45.53</td><td>33.67</td><td>10.02</td><td>11.92</td><td>21.42</td><td>6.711</td><td>17.02</td><td>10.17</td></tr><tr><td>Moonshot-v1</td><td>43.40</td><td>21.70</td><td>-</td><td>6.35</td><td>10.01</td><td>-</td><td>7.69</td><td>15.29</td><td>-</td></tr><tr><td colspan="10">Open-source VLM Models + Grounding SAM</td></tr><tr><td>Qwen-VL-Max</td><td>39.28</td><td>19.81</td><td>53.74</td><td>18.26</td><td>11.48</td><td>18.59</td><td>22.42</td><td>18.96</td><td>29.46</td></tr><tr><td>Qwen-VL-Plus</td><td>11.93</td><td>0.00</td><td>1.85</td><td>0.00</td><td>0.00</td><td>6.16</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>InternVL-3</td><td>0.00</td><td>9.34</td><td>7.44</td><td>0.00</td><td>0.00</td><td>1.15</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>InternVL-2.5</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>13.05</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td colspan="10">Geometry based methods</td></tr><tr><td>NeRF-HuGS</td><td>13.43</td><td>15.04</td><td>8.47</td><td>7.89</td><td>6.65</td><td>8.43</td><td>14.95</td><td>4.25</td><td>6.70</td></tr><tr><td>VGGT Atten.</td><td>24.14</td><td>46.37</td><td>39.41</td><td>8.73</td><td>11.04</td><td>12.86</td><td>19.15</td><td>6.69</td><td>9.28</td></tr><tr><td>VGGT Atten. + VLM</td><td>56.14</td><td>60.78</td><td>60.82</td><td>30.18</td><td>29.36</td><td>25.15</td><td>37.65</td><td>23.65</td><td>29.08</td></tr></table>

Fig. 4: Quantitative and qualitative evaluation of transient mask generation. Our method consistently outperforms baseline approaches, delivering more reliable and stable mask predictions.

## A. Mask Prior Results

Fig. 4 presents the evaluation results for prior mask generation. Overall, open-source VLMs underperform compared to their closed-source counterparts on our task. Among the closed-source models, GPT-4.1 and Claude demonstrate the best performance, suggesting a stronger grasp of the task requirements. In contrast, the VLM + Grounding DINO pipeline follows a cascaded approach, which is prone to error accumulation. NeRF-HuGS, which relies on COLMAP to infer static regions, often overpredicts and generally yields suboptimal results under sparse-view conditions. While VGGT attention matching alone significantly outperforms NeRF-HuGS, it still struggles in large textureless or poorly observed regions. In contrast, our methodâcombining VGGT with VLM guidanceâconsistently delivers the most stable and accurate results across all cases.

<!-- image-->

<table><tr><td rowspan="2">Method</td><td colspan="2">PSNRâ</td><td rowspan="2">R*</td><td rowspan="2">SSIMâ R*+BA Full</td><td rowspan="2">R*</td><td rowspan="2">LPIPSâ R*+BA Full</td></tr><tr><td>R* R*+BA</td><td>Full</td></tr><tr><td>Yoda</td><td>|24.52</td><td>26.16 26.55</td><td>|0.84</td><td>0.80</td><td>0.84| |0.24</td><td>0.26 0.23</td></tr><tr><td>Crab2</td><td>21.22</td><td>21.67 21.58</td><td>0.73</td><td>0.76</td><td>0.76 0.32</td><td>0.29 0.29</td></tr><tr><td>Corner</td><td>14.62</td><td>14.98 15.21</td><td>0.43</td><td>0.47</td><td>0.48 |0.47</td><td>0.44 0.43</td></tr><tr><td>Patio-high</td><td>13.92</td><td>13.75 14.33</td><td>0.32</td><td>0.29</td><td>0.34 0.52</td><td>0.53 0.50</td></tr></table>

Fig. 5: Quantitative and qualitative ablations of each component. We present the raw distractor masks generated by different ablation variants to highlight the effectiveness of our mask-prior-guided warm-up strategy.

<!-- image-->  
Fig. 6: Illustration of VGGT combined with Bundle Adjustment (BA). The first row shows front views, while the second row presents birdâs-eye views. Our mask priors significantly enhance the robustness and reliability of BA + Alignment process.

## B. Sparse-veiw Distractor-free 3DGS Results

Comparison on the RobustNeRF dataset. As shown in Table I, RobustGS performs the worst among all baseline methods, and WildGaussians also show relatively weak results. SpotLessSplat and Desplat are built on more advanced Gaussian Splatting frameworks with extra tailored designs and generally achieve better performance, whereas RobustGS is based on the original GS implementation. It also should be noted that the initial parameters of baselines are drawn from the original densely captured datasets. Despite this, our improved variant (RobustGS\* + Ours) achieves competitive results, enhancing RobustGS with 2â4 dB PSNR gains and RobustGS\* with 1-4 dB gains. This explicitly proves that while VGGT provides a necessary starting point, the core enhancements in rendering quality and transient removal are fundamentally driven by our high-quality mask priors and the effective warm-up strategy.

TABLE I: Quantitative results on the RobustNeRF dataset, evaluated by PSNRâ, with 4, 6, 8 training views. Best results are highlighted as first , second and third . \* means with the VGGT initialization.
<table><tr><td>Methods</td><td>4 views</td><td>Android 6 views</td><td>8 views</td><td>4 views</td><td>Crab1 6 views</td><td>8 views</td><td>4 views</td><td>Crab2 6 views</td><td>8 views</td><td>4 views</td><td>Statue 6 views</td><td>8 views</td><td>4 views</td><td>Yoda 6 views</td><td>8 views</td></tr><tr><td>WildGaussians</td><td>16.58</td><td>14.46</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>21.08</td></tr><tr><td>SpotLessSplat</td><td>119.69</td><td>17.69</td><td>14.33 16.82</td><td>22.19 26.12</td><td>18.02 22.84</td><td>17.57 24.03</td><td>19.71 24.53</td><td>19.24 24.85</td><td>20.07 26.90</td><td>12.33 113.57</td><td>12.56 13.39</td><td>12.38 14.08</td><td>18.16 24.11</td><td>18.32 27.49</td><td>30.066</td></tr><tr><td>DeSplat</td><td>18.71</td><td>17.24</td><td>16.10</td><td>25.00</td><td>23.13</td><td>24..83</td><td>23.69</td><td>23.31</td><td>25.75</td><td>14.15</td><td>13.21</td><td>12.60</td><td>23.36</td><td>25.04</td><td>28.62</td></tr><tr><td>RobustGS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RobustGS*</td><td>13.67 17.39</td><td>14.14 15.04</td><td>9.92 14.23</td><td>19.04 22.35</td><td>17.76 18.71</td><td>17.19 18.27</td><td>18.64 20.20</td><td>18.57 19.49</td><td>20.48 20.60</td><td>10.67 1347</td><td>12.26 13.42</td><td>11.64 13.04</td><td>21.72 20.04</td><td>14.88 19.39</td><td>23.96 21.32</td></tr><tr><td>RobustGS* + ours</td><td>18.00</td><td>15.85</td><td>14.88</td><td>23.81</td><td>20.23</td><td>18.99</td><td>23.17</td><td>211.51</td><td>221.58</td><td>114.38</td><td>14.4.25</td><td>14.37</td><td>22.36</td><td>24.99</td><td>26.55</td></tr></table>

TABLE II: Quantitative results on the NeRF On-the-go dataset, evaluated by PSNRâ, with 4, 6, 8 training views.
<table><tr><td rowspan="2">Methods</td><td colspan="2">Corner</td><td colspan="2"></td><td colspan="2">Fountain</td><td colspan="2">Mountain</td><td colspan="2"></td><td colspan="2"></td><td colspan="2">Patio</td><td colspan="2"></td><td colspan="2">Patio-high</td></tr><tr><td>4 views</td><td>6 views</td><td>8 views</td><td>4 views</td><td>6 views</td><td>8 views</td><td>4 views</td><td>6 views</td><td>8 views</td><td>4 views</td><td>6 views</td><td>8 views</td><td>4 views</td><td>6 views</td><td>8 views</td><td>4 views</td><td>6 views</td><td>8 views</td></tr><tr><td>WildGaussians</td><td>12.31</td><td>14.48</td><td>14.55</td><td>8.88</td><td>12.86</td><td>10.05</td><td>13.38</td><td>12.82</td><td>12.74</td><td>14.50</td><td>14.83</td><td>13.75</td><td>12.73</td><td>12.03</td><td>13.29</td><td>11.47</td><td>12.34</td><td>11.68</td></tr><tr><td>SpotLessSplat</td><td>14.14</td><td>16.68</td><td>17.05</td><td>9.34</td><td>13.43</td><td>10.89</td><td>14.70</td><td>13.99</td><td>14.64</td><td>14.80</td><td>16.14</td><td>15.06</td><td>14.05</td><td>13.36</td><td>13.97</td><td>12.39</td><td>12.79</td><td>12.67</td></tr><tr><td>DeSplat</td><td>14.35</td><td>15.97</td><td>16.89</td><td>10.96</td><td>13.90</td><td>11.54</td><td>13.58</td><td>13.86</td><td>13.71</td><td>15.25</td><td>16.64</td><td>15.42</td><td>15.14</td><td>14.13</td><td>16.10</td><td>12.97</td><td>14.05</td><td>13.58</td></tr><tr><td>RobustGS</td><td>12.17</td><td>14.06</td><td>14.62</td><td>8.00</td><td>11.17</td><td>9.94</td><td>11.14</td><td>10.07</td><td>10.85</td><td>14.27</td><td>13.90</td><td>15.00</td><td>12.94</td><td>11.74</td><td>14.18</td><td>10.43</td><td>12.68</td><td>11.53</td></tr><tr><td>RobustGS*</td><td>13.61</td><td>13.82</td><td>13.89</td><td>10.92</td><td>13.04</td><td>11.41</td><td>14.07</td><td>13.16</td><td>13.42</td><td>14.35</td><td>14.45</td><td>14.91</td><td>12.25</td><td>11.73</td><td>12.05</td><td>12.78</td><td>12.44</td><td>12.17</td></tr><tr><td>RobustGS* + ours</td><td>14.94</td><td>14.76</td><td>15.21</td><td>11.54</td><td>13.79</td><td>12.82</td><td>14.74</td><td>14.16</td><td>14.70</td><td>15.20</td><td>15.82</td><td>16.02</td><td>12.73</td><td>11.67</td><td>13.09</td><td>13.87</td><td>14.61</td><td>14.33</td></tr></table>

Comparison on the NeRF on-the-go dataset. Similar to the trend observed on the RobustNeRF dataset, our full pipeline (RobustGS\* + Ours) consistently surpasses the RobustGS and RobustGS\*. As shown in Table II, our method achieves the best performance in 10 out of 18 cases, outperforming all baselines. More importantly, as demonstrated in Fig. 3, our robust mask priors and warm-up strategy consistently and effectively eliminate most transient objects across diverse, unconstrained scenes, proving the effectiveness and generalization ability of our method.

Ablation studies. We conduct ablation studies on four scenes (Yoda, Crab2, Corner, and Spot), each with 8 input views. We compare three variants: RobustGS\* (R\*), RobustGS\* with GSBA $( R ^ { * } { + } B A )$ , and our full model $( R ^ { * } { + } B A { + } W )$ . The results are shown in Fig. 5. R\*+BA consistently outperforms R\*, highlighting the importance of incorporating GSBA. Our full model achieves slightly better overall performance than R\*+BA and demonstrates a clear advantage in mask generation during GS training. The qualitative parts of Fig. 5 can better validate this advantage. The key difference lies in our mask-prior-guided warm-up strategy, which effectively reduces the burden on the mask predictor in RobustGS.

Discussion. We also investigate the Bundle Adjustments before the GS training. VGGT also demonstrates that refining predicted camera poses and tracks with BA can further improve accuracy. However, the predicted depth maps are not aligned with the refined extrinsics and intrinsics. To address this, we utilize the sparse points obtained from the BA refinement stage as pseudo-ground truth, aligning the predicted depth maps to these points using a RANSACbased linear regression model. This alignment process is highly sensitive, particularly in scenes containing multiple transients. As shown in Fig. 6, incorporating our mask priors can further clearly improve the success rate, demonstrating that our approach not only aids novel view synthesis but also effectively stabilizes Bundle Adjustment under challenging sparse-view and distractor-heavy conditions.

## VI. CONCLUSION

In this work, we present a novel method for mask prior generation to enhance distractor-free 3D Gaussian Splatting (3DGS) under sparse-view conditions. Our approach leverages the geometry foundation model (i.e., VGGT) alongside powerful Vision-Language Models (VLMs) to produce robust and reliable rendering results. With a simple maskprior-guided warm-up strategy, we significantly boost the performance of existing distractor-free 3DGS methods.

## REFERENCES

[1] S. Zhu, L. Mou, D. Li, B. Ye, R. Huang, and H. Zhao, âVrrobo: A real-to-sim-to-real framework for visual robot navigation and locomotion,â IEEE Robotics and Automation Letters, 2025.

[2] Z. Xu, R. Jin, K. Wu, Y. Zhao, Z. Zhang, J. Zhao, F. Gao, Z. Gan, and W. Ding, âHgs-planner: Hierarchical planning framework for active scene reconstruction using 3d gaussian splatting,â in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 14 161â14 167.

[3] Y. Liu, R. Ni, and A. H. Qureshi, âPhysics-informed neural mapping and motion planning in unknown environments,â IEEE Transactions on Robotics, 2025.

[4] R. Jin, Y. Gao, Y. Wang, Y. Wu, H. Lu, C. Xu, and F. Gao, âGs-planner: A gaussian-splatting-based planning framework for active high-fidelity reconstruction,â in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 11 202â11 209.

[5] K. Yamazaki, T. Hanyu, K. Vo, T. Pham, M. Tran, G. Doretto, A. Nguyen, and N. Le, âOpen-fusion: Real-time open-vocabulary 3d mapping and queryable scene representation,â in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 9411â9417.

[6] H. Jun, H. Yu, and S. Oh, âRenderable street view map-based localization: Leveraging 3d gaussian splatting for street-level positioning,â in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 5635â5640.

[7] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[8] N. Max, âOptical models for direct volume rendering,â IEEE Transactions on visualization and computer graphics, vol. 1, no. 2, pp. 99â108, 2002.

[9] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[10] J. Zhang, J. Li, X. Yu, L. Huang, L. Gu, J. Zheng, and X. Bai, âCor-gs: sparse-view 3d gaussian splatting via co-regularization,â in European Conference on Computer Vision. Springer, 2024, pp. 335â352.

[11] J. Li, J. Zhang, X. Bai, J. Zheng, X. Ning, J. Zhou, and L. Gu, âDngaussian: Optimizing sparse-view 3d gaussian radiance fields with global-local depth normalization,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 20 775â20 785.

[12] Y. Chen, H. Xu, C. Zheng, B. Zhuang, M. Pollefeys, A. Geiger, T.-J. Cham, and J. Cai, âMvsplat: Efficient 3d gaussian splatting from sparse multi-view images,â in European Conference on Computer Vision. Springer, 2024, pp. 370â386.

[13] B. Ye, S. Liu, H. Xu, X. Li, M. Pollefeys, M.-H. Yang, and S. Peng, âNo pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images,â in The Thirteenth International Conference on Learning Representations, 2025.

[14] S. Zhang, J. Wang, Y. Xu, N. Xue, C. Rupprecht, X. Zhou, Y. Shen, and G. Wetzstein, âFlare: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views,â in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[15] S. Sabour, S. Vora, D. Duckworth, I. Krasin, D. J. Fleet, and A. Tagliasacchi, âRobustnerf: Ignoring distractors with robust losses,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 20 626â20 636.

[16] J. Chen, Y. Qin, L. Liu, J. Lu, and G. Li, âNerf-hugs: Improved neural radiance fields in non-static scenes using heuristics-guided segmentation,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19 436â19 446.

[17] J. L. Schonberger and J.-M. Frahm, âStructure-from-motion revisited,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 4104â4113.

[18] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, âVggt: Visual geometry grounded transformer,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 5294â5306.

[19] P. Ungermann, A. Ettenhofer, M. NieÃner, and B. Roessle, âRobust 3d gaussian splatting for novel view synthesis in presence of distractors,â in DAGM German Conference on Pattern Recognition. Springer, 2024, pp. 153â167.

[20] R. Martin-Brualla, N. Radwan, M. S. Sajjadi, J. T. Barron, A. Dosovitskiy, and D. Duckworth, âNerf in the wild: Neural radiance fields for unconstrained photo collections,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 7210â7219.

[21] W. Ren, Z. Zhu, B. Sun, J. Chen, M. Pollefeys, and S. Peng, âNerf onthe-go: Exploiting uncertainty for distractor-free nerfs in the wild,â in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[22] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al., âDinov2: Learning robust visual features without supervision,â Transactions on Machine Learning Research Journal, pp. 1â31, 2024.

[23] S. Sabour, L. Goli, G. Kopanas, M. Matthews, D. Lagun, L. Guibas, A. Jacobson, D. Fleet, and A. Tagliasacchi, âSpotlesssplats: Ignoring distractors in 3d gaussian splatting,â ACM Transactions on Graphics, vol. 44, no. 2, pp. 1â11, 2025.

[24] J. Lin, J. Gu, L. Fan, B. Wu, Y. Lou, R. Chen, L. Liu, and J. Ye, âHybridgs: Decoupling transients and statics with 2d and 3d gaussian splatting,â in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[25] Y. Wang, M. Klasson, M. Turkulainen, S. Wang, J. Kannala, and A. Solin, âDeSplat: Decomposed Gaussian splatting for distractor-free rendering,â in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[26] C. Fu, Y. Zhang, K. Yao, G. Chen, Y. Xiong, C. Huang, S. Cui, and X. Cao, âRobustsplat: Decoupling densification and dynamics for transient-free 3dgs,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 27 126â27 136.

[27] C. Fu, G. Chen, Y. Zhang, K. Yao, Y. Xiong, C. Huang, S. Cui, Y. Matsushita, and X. Cao, âRobustsplat++: Decoupling densification, dynamics, and illumination for in-the-wild 3dgs,â arXiv preprint arXiv:2512.04815, 2025.

[28] R. Wang, Q. Lohmeyer, M. Meboldt, and S. Tang, âDegauss: Dynamicstatic decomposition with gaussian splatting for distractor-free 3d

reconstruction,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 6294â6303.

[29] J. Kulhanek, S. Peng, Z. Kukelova, M. Pollefeys, and T. Sattler, âWildGaussians: 3D gaussian splatting in the wild,â NeurIPS, 2024.

[30] W. Park, M. Nam, S. Kim, S. Jo, and S. Lee, âForestsplats: Deformable transient field for gaussian splatting in the wild,â arXiv preprint arXiv:2503.06179, 2025.

[31] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, âHigh-resolution image synthesis with latent diffusion models,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684â10 695.

[32] D. Zhang, C. Wang, W. Wang, P. Li, M. Qin, and H. Wang, âGaussian in the wild: 3d gaussian splatting for unconstrained image collections,â in European Conference on Computer Vision. Springer, 2024, pp. 341â359.

[33] J. Xu, Y. Mei, and V. Patel, âWild-gs: Real-time novel view synthesis from unconstrained photo collections,â Advances in Neural Information Processing Systems, vol. 37, pp. 103 334â103 355, 2024.

[34] L. Goli, S. Sabour, M. Matthews, M. Brubaker, D. Lagun, A. Jacobson, D. J. Fleet, S. Saxena, and A. Tagliasacchi, âRomo: Robust motion segmentation improves structure from motion,â arXiv preprint arXiv:2411.18650, 2024.

[35] A. Markin, V. Pryadilshchikov, A. Komarichev, R. Rakhimov, P. Wonka, and E. Burnaev, âT-3dgs: Removing transient objects for 3d scene reconstruction,â arXiv preprint arXiv:2412.00155, 2024.

[36] K. Xu, T. H. E. Tse, J. Peng, and A. Yao, âDas3r: Dynamicsaware gaussian splatting for static scene reconstruction,â arXiv preprint arXiv:2412.19584, 2024.

[37] Z. Fan, K. Wen, W. Cong, K. Wang, J. Zhang, X. Ding, D. Xu, B. Ivanovic, M. Pavone, G. Pavlakos, Z. Wang, and Y. Wang, âInstantsplat: Sparse-view gaussian splatting in seconds,â 2024.

[38] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, âDust3r: Geometric 3d vision made easy,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 697â20 709.

[39] V. Leroy, Y. Cabon, and J. Revaud, âGrounding image matching in 3d with mast3r,â in European Conference on Computer Vision. Springer, 2024, pp. 71â91.

[40] Y. Li, X. Wang, J. Wu, Y. Ma, and Z. Jin, âSparsegs-w: Sparse-view 3d gaussian splatting in the wild with generative priors,â arXiv preprint arXiv:2503.19452, 2025.

[41] X. Chen, Y. Chen, Y. Xiu, A. Geiger, and A. Chen, âEasi3r: Estimating disentangled motion from dust3r without training,â arXiv preprint arXiv:2503.24391, 2025.

[42] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Å. Kaiser, and I. Polosukhin, âAttention is all you need,â Advances in neural information processing systems, vol. 30, 2017.

[43] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, et al., âSegment anything,â in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4015â4026.

[44] L. Qi, J. Kuen, T. Shen, J. Gu, W. Li, W. Guo, J. Jia, Z. Lin, and M.-H. Yang, âHigh quality entity segmentation,â in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 4024â4033.

[45] T. Otonari, S. Ikehata, and K. Aizawa, âEntity-nerf: Detecting and removing moving entities in urban scenes,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 892â20 901.

[46] T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan, Z. Zeng, H. Zhang, F. Li, J. Yang, H. Li, Q. Jiang, and L. Zhang, âGrounded sam: Assembling open-world models for diverse visual tasks,â 2024.

[47] Q. Guo, S. De Mello, H. Yin, W. Byeon, K. C. Cheung, Y. Yu, P. Luo, and S. Liu, âRegiongpt: Towards region understanding vision language model,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 13 796â13 806.

[48] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al., âChain-of-thought prompting elicits reasoning in large language models,â Advances in neural information processing systems, vol. 35, pp. 24 824â24 837, 2022.

[49] Q. Wu, X. Yang, Y. Zhou, C. Fang, B. Song, X. Sun, and R. Ji, âGrounded chain-of-thought for multimodal large language models,â arXiv preprint arXiv:2503.12799, 2025.