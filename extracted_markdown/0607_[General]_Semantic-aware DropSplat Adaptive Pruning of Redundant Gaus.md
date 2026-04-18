# Semantic-aware DropSplat: Adaptive Pruning of Redundant Gaussians for 3D Aerial-View Segmentation

Xu Tang1\*, Junan Jia1, Yijing Wang1, Jingjing Ma1, Xiangrong Zhang1

1School of Artificial Intelligence,

Xidian University, Xiâan, China

tangxu128@gmail.com, 24171214056@stu.xidian.edu.cn, yijingwang@stu.xidian.edu.cn, smallpig32@sina.com, xrzhang@mail.xidian.edu.cn

## Abstract

In the task of 3D Aerial-view Scene Semantic Segmentation (3D-AVS-SS), traditional methods struggle to address semantic ambiguity caused by scale variations and structural occlusions in aerial images. This limits their segmentation accuracy and consistency. To tackle these challenges, we propose a novel 3D-AVS-SS approach named SAD-Splat. Our method introduces a Gaussian point drop module, which integrates semantic confidence estimation with a learnable sparsity mechanism based on the Hard Concrete distribution. This module effectively eliminates redundant and semantically ambiguous Gaussian points, enhancing both segmentation performance and representation compactness. Furthermore, SAD-Splat incorporates a high-confidence pseudolabel generation pipeline. It leverages 2D foundation models to enhance supervision when ground-truth labels are limited, thereby further improving segmentation accuracy. To advance research in this domain, we introduce a challenging benchmark dataset: 3D Aerial Semantic (3D-AS), which encompasses diverse real-world aerial scenes with sparse annotations. Experimental results demonstrate that SAD-Splat achieves an excellent balance between segmentation accuracy and representation compactness. It offers an efficient and scalable solution for 3D aerial scene understanding.

## Introduction

As a branch of 3D multi-view semantic segmentation, 3D Aerial-view Scene Semantic Segmentation (3D-AVS-SS) plays a crucial role in various remote sensing (RS) applications, including land use monitoring, urban planning, and disaster response (Huang et al. 2023; Rahnemoonfar, Chowdhury, and Murphy 2022). Its goal is to assign semantic labels to each pixel in multi-view aerial images captured from high-altitude perspectives around a target scene. Compared to the normal 2D images, aerial images often suffer from significant scale variations and structural occlusions, which hinder spatial consistency and semantic alignment across multiple views. Therefore, constructing a unified 3D scene representation and performing semantic reasoning in 3D multi-view aerial images has become a promising research topic for achieving more accurate and consistent segmentation results.

There are two steps in the standard 3D multi-view semantic segmentation process. The first step is to reconstruct the 3D scene using the multi-view images. The second step is to embed the semantic features into the obtained 3D scene and render the semantic 3D scene back into 2D views for semantic predictions. In the 3D reconstruction step, Neural Radiance Fields (NeRF) (Mildenhall et al. 2021) and 3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) are two popular tools that have gained significant attention in recent years. NeRF introduces an implicit volumetric representation by learning density and color fields from multi-view images, enabling photorealistic rendering. In contrast, 3DGS employs an explicit and structured modeling paradigm based on parameterized 3D Gaussians, which jointly encode geometry and appearance. It is gaining increasing attention due to its rendering efficiency and structural interpretability. For this reason, our work is also built upon 3DGS. For the semantic segmentation step, powerful 2D foundation models (e.g., CLIP (Radford et al. 2021) and SAM (Kirillov et al. 2023)) are first leveraged to extract semantic knowledge, which is subsequently distilled into the 3D representation. Semantic segmentation is then performed by rendering the 3D scene onto 2D images. Based on the basic techniques mentioned above, many successful 3D multi-view semantic segmentation models have been developed (Zhou et al. 2024; Cen et al. 2025).

Although the above methods perform well, applying them to 3D-AVS-SS tasks directly still presents many challenges. In the 3D reconstruction step, since manual annotation of aerial images is scarce, semantically ambiguous regions often appear. Within the 3DGS framework, the model tends to frequently duplicate or split Gaussian points in uncertain areas, producing dense, redundant, and noisy semantic representations. This ultimately undermines the spatial consistency and reduces the overall compactness of the results. In the semantic segmentation step, as the prevalent 2D foundation models are always pre-trained by natural images, the semantic knowledge transferred from them for multiview aerial images often contains noise and inconsistencies, which lead to the problems of category confusion, misclassification, and inter-view disagreement. Taking Fig. 1a as an example, the left side presents a natural scene (âbedâ) (Liu et al. 2023) with its corresponding RGB image, ground truth (GT) segmentation, and the prediction from a 2D foundation model. It can be observed that the model performs reasonably well in this setting. In contrast, the right side shows an example from an aerial scene (âcityâ). Compared to the GT, the prediction from the 2D foundation model contains numerous errors; for example, roads and trees are rarely correctly classified, resulting in a large amount of noisy supervision.

<!-- image-->

<!-- image-->  
Figure 1: (a) Comparison of segmentation results between natural and aerial images using foundation models. (b) Effectiveness of the semantic-aware drop. The proposed drop mechanism significantly reduces the number of Gaussian points while maintaining segmentation accuracy, leading to a more compact and efficient model representation.

To address the above challenges, we propose a new 3D-AVS-SS model named Semantic-aware DropSplat (SAD-Splat) under the framework of 3DGS. To tackle the issue of excessive redundant and semantically ambiguous Gaussian points generated during the 3D reconstruction process in 3DGS, SAD-Splat introduces a learnable drop strategy. This strategy estimates the retention probability of each Gaussian point by integrating a base drop rate, semantic confidence estimation, and a sparsity constraint based on the hard-concrete (Louizos, Welling, and Kingma 2017) distribution. Also, it effectively suppresses Gaussian points that are semantically ambiguous or contribute little, resulting in a more efficient and compact model representation, the effect of applying this strategy to the same scene is shown in Fig. 1b. In addition to tackling the dissatisfactory performance of 2D foundation models on multi-view aerial images, SAD-Splat incorporates a filtering mechanism that only utilizes semantic information with sufficiently high confidence for supervision, thereby minimizing the influence of semantically uncertain regions.

To advance research in 3D-AVS-SS, we construct a new dataset named 3D-Aerial Semantic (3D-AS). It contains diverse real-world aerial scenes with pixel-level annotations and poses challenges such as limited supervision, class imbalance, and scene diversity.

## Related Work

In this work, our core pipeline leverages pre-trained foundation models to extract semantic features from images, which are subsequently transferred to 3D space via knowledge distillation. Based on this technical framework, we review related work from two primary perspectives. First, we discuss foundation models widely employed in image segmentation tasks, encompassing vision-language models and large-scale segmentation networks. Second, we summarize recent advancements in embedding semantic features within the 3D Gaussian Splatting (3DGS) framework.

Foundation Models for Segmentation In recent years, the development of foundation models has significantly advanced semantic segmentation. These models can be broadly categorized into two groups. The first group comprises vision-language models. CLIP achieves open-vocabulary recognition by learning joint image-text embeddings from large-scale datasets; OpenCLIP (Cherti et al. 2023), as an optimized open-source version, enhances training efficiency and applicability. GeoRSCLIP (Zhang et al. 2024) aligns aerial images with geographic semantics through contrastive learning, demonstrating superior performance in land cover classification and scene understanding. The second group focuses on segmentation models, such as DINO (Caron et al. 2021) and CLIP-based approaches like MaskCLIP (Dong et al. 2023) and LSeg (Li et al. 2022), which integrate crossmodal semantic understanding with dense spatial prediction. Additionally, segmentation models based on the SAM and its latest version, SAM2 (Ravi et al. 2024), trained on massive datasets, exhibit strong generalization and zero-shot capabilities.

However, these models still face limitations in 3D-AVS-SS. Insufficient exploitation of geometric relationships across views and domain gaps results in degraded performance.

Semantic Segmentation with 3D Gaussian Splatting Recent work, such as feature-3DGS (Zhou et al. 2024), distills features from large 2D foundation models into 3DGS, enabling semantic rendering and open-vocabulary segmentation. LangSplat (Qin et al. 2024) introduces language supervision to build 3D semantic fields that support precise spatial queries. OpenSplat3D (Piekenbrinck et al. 2025) associates semantic information with Gaussian points, leveraging SAM masks and contrastive loss combined with vision-language embeddings to achieve open-vocabulary 3D instance segmentation. This approach demonstrates strong cross-view consistency and high-accuracy segmentation across multiple datasets, significantly advancing the application of 3DGS for semantic segmentation.

Despite these advances, existing methods still face challenges such as oversaturation of Gaussian points and semantic confusion. To address these issues, we propose a semantic-aware drop mechanism within the 3DGS framework. This mechanism jointly prunes redundant Gaussian points and enhances semantic compactness, thus improving the 3D-AVS-SS performance.

## Method

The overall pipeline of SAD-Splat is illustrated in Fig. 2. Given multi-view aerial images, a small number of groundtruth semantic labels, and corresponding text descriptions, the preprocessing stage generates high-confidence pseudolabels and semantic confidence maps. During training, the system jointly reconstructs semantic features and learns both the semantic confidence and the learnable drop rate for each Gaussian point. This joint optimization integrates semantic importance and structural redundancy, ensuring both semantic accuracy and model compactness.

## Preliminary

3DGS (Kerbl et al. 2023) is an efficient method for representing complex 3D scenes using a collection of anisotropic Gaussian primitives. Each Gaussian simultaneously encodes both geometric and appearance-related information, including spatial position, color, opacity, rotation, and scale, enabling a unified representation of scene structure and visual content.

Given a set of multi-view images with camera poses, 3DGS initializes a set of Gaussians as follows:

$$
G = \{ g _ { i } \} _ { i = 1 } ^ { N } ,
$$

where each Gaussian gi is defined as a five-tuple:

$$
g _ { i } = \{ \mu _ { i } , c _ { i } , \alpha _ { i } , R _ { i } , s _ { i } \} ,
$$

where $\mu _ { i } ~ \in ~ \mathbb { R } ^ { 3 }$ denotes the 3D center, $c _ { i } ~ \in ~ \mathbb { R } ^ { 3 }$ represents the RGB color, $\alpha _ { i } ~ \in ~ [ 0 , 1 ]$ indicates the opacity, $R _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ implies the rotation matrix, and $s _ { i } \in \mathbb { R } ^ { 3 }$ refers to the anisotropic scaling factors.During the rendering process, Gaussians are projected onto the image plane and undergo front-to-back alpha compositing based on view-dependent opacity to achieve proper visibility ordering.

Based on 3DGS, Feature 3DGS (Zhou et al. 2024) introduces a learnable semantic embedding vector $f _ { i } \in \mathbb { R } ^ { d }$ for each Gaussian, enabling dense semantic modeling in 3D space. These semantic features are aligned with a 2D foundation model through distillation to obtain cross-modal supervision signals. In the semantic rendering stage, the semantic descriptor $S ( p )$ for a pixel p is computed by weighted aggregation of the semantic features of its visible Gaussian set ${ \mathit { g _ { p } } } ,$ where both opacity and visibility factors determine the weights. Through this mechanism, semantic information is effectively incorporated into the 3DGS framework, enabling semantic segmentation of 3D scenes based on Gaussian representations. Our SAD-Splat is developed based on Feature 3DGS.

## Gaussian Point Drop Implementation

During the training stage of Feature 3DGS, regions with semantic ambiguity or conflict often accumulate large gradients. These large gradients can trigger density control operations, such as splitting or cloning, which may lead to the generation of redundant and semantically ambiguous regions. To address this issue, we propose a drop operation specifically targeting these ambiguous regions to enhance the efficiency and stability of SAD-Splat. The drop operation is designed by comprehensively considering both semantic and structural factors.

## Semantic Confidence Drop Module

To evaluate the semantic importance of Gaussian points in 3DGS, we generate a confidence map using SAM (Kirillov et al. 2023) and GeoRSCLIP (Zhang et al. 2024). SAM segments training images to extract object regions and their edge information. Meanwhile, GeoRSCLIPâs image encoder derives semantic feature vectors for objects, and its text encoder extracts semantic vectors for category descriptions. Cosine similarities between these vectors yield per-category similarity scores, which are mapped to image coordinates to form a confidence map Confidence $\in \mathbb { R } ^ { H \times W \times C }$ , where H and W denote image height and width, and C means the number of categories. For each rendered image, the 3D centers of visible Gaussian points are projected onto the confidence map, and the maximum score across the category dimension, weighted by the pointâs opacity, is extracted as the view-specific confidence. For multi-view visible points, the average confidence across views is computed to assess their semantic importance in the scene. Semantically ambiguous regions typically exhibit lower confidence scores.

## Learnable Structure Drop Module

To identify Gaussian points in the model structure that can be dropped, we utilize the Hard Concrete distribution (Louizos, Welling, and Kingma 2017) to obtain a differentiable approximation of binary drop, thereby optimizing Gaussian point pruning in 3DGS. Each Gaussian point is assigned a learnable parameter log Î±, which governs the drop probability in logarithmic form to facilitate gradientbased optimization. A binary mask is sampled using noiseaugmented logits and a bounded sigmoid function, enabling end-to-end training. To promote sparsity, we introduce an $\mathcal { L } _ { 0 }$ regularization term to penalize the expected number of active Gaussian points. The non-zero activation probability is defined as:

$$
P _ { \mathrm { n o n z e r o } } = \sigma \left( { \frac { \log \alpha - \tau \cdot \log \left( { \frac { \mathrm { t h r e s h o l d } } { 1 - \mathrm { t h r e s h o l d } } } \right) } { \tau } } \right) ,
$$

where log Î± is the learnable parameter controlling retention, Ï is the temperature parameter adjusting distribution smoothness, and threshold $= \ { \frac { - l } { r - l } }$ is derived from Hard Concrete bounds l and r. Then, the regularization loss can be defined as:

$$
\mathcal { L } _ { L _ { 0 } } = \mathbb { E } [ P _ { \mathrm { n o n z e r o } } ] .
$$

<!-- image-->  
Figure 2: Overview of SAD-Splat. The upper part shows the preprocessing pipeline, where SAM, GeoRSCLIP, and a highconfidence filtering strategy are used to generate semantic feature maps and confidence maps. The right part depicts the training process, in which a drop operation is performed after a fixed number of training iterations and is repeated throughout training. The drop operation is guided by the drop rate produced by the bottom module, which integrates a base drop rate, semantic confidence, and a learnable drop parameter to estimate a drop probability for each Gaussian point.

It minimizes the expected non-zero probability, effectively reducing redundant Gaussian points.

After N training iterations, we integrate the base drop rate $P _ { \mathrm { b a s e } } ,$ the average semantic confidence $P _ { \mathrm { c o n f i d e n c e } } ,$ and the learnable drop rate $P _ { \mathrm { l e a r n e d } }$ to compute the comprehensive drop probability for each Gaussian point

$$
P _ { \mathrm { d r o p } } = P _ { \mathrm { b a s e } } \cdot ( 1 - P _ { \mathrm { c o n f i d e n c e } } ) \cdot P _ { \mathrm { l e a r n e d } } .
$$

Here, $P _ { \mathrm { b a s e } }$ controls the overall drop scale, $P _ { \mathrm { c o n f i d e n c e } }$ reflects semantic importance, and $P _ { \mathrm { l e a r n e d } }$ enables adaptive adjustment. Periodic pruning is performed every N iterations to remove semantically ambiguous or structurally invalid Gaussian points, effectively suppressing redundant regions.

## Pseudo-Label Generation Module

In 3D-AVS-SS tasks, the absence of ground-truth labels leads to inadequate supervision. This challenge is particularly evident for views where annotations for targets, such as buildings, are missing. To address this issue, we introduce a preprocess stage where 2D foundation models are leveraged to generate pseudo-labels, providing auxiliary supervision signals.

We reuse the confidence map generation process to compute similarity scores between the image-derived features of object regions and the text-derived features of category descriptions. To ensure the reliability of supervision signals, three metrics are computed for each target: the TOP1 score (the similarity with the most likely class), the difference between TOP1 and TOP2 scores (denoted as $\Delta T O P .$ , represents the modelâs confidence dominance in favor of the predicted category), and entropy (indicating the uncertainty of the semantic distribution). The mean and standard deviation for each metric are calculated over all target samples in the training dataset to reflect the current data distribution. Only targets satisfying the following conditions are retained:

$$
\left\{ \begin{array} { l l } { T O P 1 > m e a n ( T O P 1 ) + s t d ( T O P 1 ) , } \\ { \Delta T O P > m e a n ( \Delta T O P ) + s t d ( \Delta T O P ) , } \\ { E n t r o p y < m e a n ( E n t r o p y ) + s t d ( E n t r o p y ) . } \end{array} \right.
$$

These conditions filter out targets with low confidence or high ambiguity, ensuring robust pseudo-labels. The features of selected targets are embedded back into their original image positions to construct a pseudo-feature map. Non-target regions are assigned zero values and masked to exclude them from gradient computations, thereby focusing optimization on reliable semantic regions.

## Training Loss

The training loss comprises a semantic feature loss and an RGB reconstruction loss. The semantic feature loss encourages alignment between the predicted feature map and the ground-truth feature map by maximizing the cosine similarity at each pixel. It is defined as:

$$
\mathcal { L } _ { \mathrm { s e m a n t i c } } = 1 - \frac { 1 } { | \Omega | } \sum _ { ( x , y ) \in \Omega } \frac { f _ { \mathrm { p r e d } } ( x , y ) \cdot f _ { \mathrm { g t } } ( x , y ) } { \| f _ { \mathrm { p r e d } } ( x , y ) \| _ { 2 } \cdot \| f _ { \mathrm { g t } } ( x , y ) \| _ { 2 } } ,
$$

where $f _ { p r e d }$ is the predicted feature at pixel $( x , y ) , f _ { g t }$ is the ground-truth feature, and â¦ denotes the set of all pixel positions.The RGB reconstruction loss, following 3D Gaussian Splatting practices, combines L1 and SSIM terms to balance pixel-level accuracy and perceptual quality:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r g b } } = ( 1 - \lambda _ { \mathrm { S S I M } } ) \cdot \| I _ { \mathrm { r e n d e r e d } } - I _ { \mathrm { i n p u t } } \| _ { 1 } } \\ { + \lambda _ { \mathrm { S S I M } } \cdot ( 1 - \mathrm { S S I M } ( I _ { \mathrm { r e n d e r e d } } , I _ { \mathrm { i n p u t } } ) ) . } \end{array}
$$

Then, the total loss function is formulated as:

$$
{ \mathcal { L } } = \lambda _ { \mathrm { s e m a n t i c } } \cdot { \mathcal { L } } _ { \mathrm { s e m a n t i c } } + \lambda _ { \mathrm { r g b } } \cdot { \mathcal { L } } _ { \mathrm { r g b } } + \lambda _ { L _ { 0 } } \cdot { \mathcal { L } } _ { L _ { 0 } } ,
$$

where $\lambda _ { \mathrm { s e m a n t i c } } , \lambda _ { \mathrm { r g b } }$ , and $\lambda _ { L _ { 0 } }$ are weighting coefficients, and $\mathcal { L } _ { L _ { 0 } }$ represents the regularization loss introduced by the Gaussian point drop module.

## Dataset

To evaluate SAD-Splat, we constructed a highly challenging dataset, named 3D-AS. The dataset comprises three scene categories sourced from Google Earth, with each category containing three scenes, resulting in a total of nine realworld sub-scenes.

To simulate surround-view capture around target scenes, we extracted screenshots from Google Earth. For each scene, approximately 70 images were captured at roughly equal angular intervals, covering diverse viewpoints, with each image having a resolution of roughly 1600 Ã 900 pixels. By leveraging Google Earthâs scale reference and the size characteristics of common objects, we estimated that each pixel corresponds to approximately 0.3â0.5 meters in the physical world. Each scene shares the same set of pixel-level label categories, covering six or more common object types. For every sub-scene, 10 images were manually annotated. Of these, 10% (7 images) were designated as the test set, with the remaining 3 images used for training. An overview of the dataset is provided in Table 1, while detailed information for each scene will be included in the supplementary material.

3D-AS presents three key challenges. First, labeled samples are scarce, with only 4.8% (3/63) of training samples annotated. Second, significant intra-class variance and interclass imbalance exist; for instance, the âBuildingâ label encompasses diverse architectural types, while images often contain abundant âGroundâ and âBuildingâ labels but minimal âGrassâ in Fig. 3. Third, the dataset includes varied real-world environments, specifically City (dense urban areas with complex structures), Country (open rural fields with sparse objects), and Port (waterfront regions with reflective surfaces and industrial facilities), each presenting distinct spatial and semantic distributions that pose significant challenges for model generalization.

## Experiments

In this section, we present the basic experimental results and analysis of SAD-Splat.

<table><tr><td>Scene</td><td>Views</td><td>Categories</td><td>Effective Pixel(%)</td></tr><tr><td>City #0</td><td>70</td><td>6</td><td>89.0</td></tr><tr><td>City #1</td><td>70</td><td>6</td><td>91.0</td></tr><tr><td>City #2</td><td>70</td><td>6</td><td>91.8</td></tr><tr><td>Country #0</td><td>70</td><td>7</td><td>95.9</td></tr><tr><td>Country #1</td><td>70</td><td>7</td><td>98.6</td></tr><tr><td>Country #2</td><td>70</td><td>7</td><td>99.8</td></tr><tr><td>Port #0</td><td>70</td><td>8</td><td>99.8</td></tr><tr><td>Port #1</td><td>70</td><td>8</td><td>97.9</td></tr><tr><td>Port #2</td><td>70</td><td>8</td><td>98.8</td></tr></table>

Table 1: Overview of 3D-AS Dataset

<!-- image-->  
Figure 3: Displays the first image of each sub-dataset in the dataset.

## Experimental Setting

All experiments were conducted on 3D-AS. To ensure reproducibility, we fixed the random seed to 42 and accordingly split the dataset into training and test sets. Specifically, images numbered 0, 7, and 28 were used as the training set, while images numbered 14, 21, 35, 42, 49, 56, and 63 were used as the test set. Performance evaluation metrics include mean Intersection over Union (mIoU) and accuracy.

## Implementation Details:

All experiments were conducted on an NVIDIA GeForce RTX 3090 GPU with 24GB of memory and PyTorch. During training, the dimensions of the images were uniformly reduced to half the width and height of the original images. We conducted experiments with 7000 training epochs across all scenes, setting the hyperparameter for the number of discarded epochs N to 500. Our method simultaneously reconstructs RGB and semantic information, with the semantic dimension set to 32.

## Comparison Experiments

To evaluate the effectiveness of SAD-Splat, we conducted a comprehensive comparison with state-of-the-art methods in two categories: (1) 2D open-vocabulary segmentation methods, including LSeg (Kerr et al. 2023) and MaskCLIP (Fu et al. 2024); (2) 3D open-vocabulary segmentation methods, including LERF (Kerr et al. 2023), LangSplat (Qin et al.

<table><tr><td rowspan="2">Method</td><td colspan="6">City</td><td colspan="6">Country</td><td colspan="6">Port</td></tr><tr><td>Scene 0 mIoU mAcc</td><td></td><td>mIoU</td><td>Scene 1 mAcc</td><td>mIoU</td><td>Scene 2 mAcc</td><td>Scene 0 mIoU mAcc</td><td></td><td>Scene 1 mIoU</td><td>mAcc</td><td>Scene 2 mIoU</td><td>mAcc</td><td>Scene 0 mIoU mAcc</td><td></td><td>Scene 1 mIoU mAcc</td><td></td><td>Scene 2 mIOU mAcc</td></tr><tr><td>LSeg</td><td>37.5</td><td>67.0</td><td>41.1</td><td>74.9</td><td>44.5</td><td>73.2</td><td>22.5</td><td>31.4</td><td>25.4</td><td>27.2</td><td>17.6</td><td>21.5</td><td>77.0</td><td>28.2</td><td></td><td>55.8</td><td>62.0</td></tr><tr><td> MaskKCLIP</td><td>33.1</td><td>49.5</td><td>28.8</td><td>44.9</td><td>37.8</td><td>55.0</td><td>23.8</td><td>33.6</td><td>30.2</td><td>52.0 21.5</td><td>41.6</td><td>45.2 46.9</td><td>76.0</td><td>35.6</td><td>63.4</td><td>24.0 31.8</td><td>67.2</td></tr><tr><td>LERF</td><td>11.7</td><td>34.5</td><td>7.2</td><td>23.3</td><td>7.5</td><td>23.8</td><td>5.5</td><td>20.2</td><td>4.0 16.3</td><td>4.1</td><td>17.6</td><td>4.5</td><td>14.7</td><td>7.6</td><td>26.5</td><td>4.6</td><td>17.7</td></tr><tr><td>LangSplat</td><td>11.2</td><td>26.6</td><td>8.3</td><td>20.3</td><td>1.7</td><td>4.1</td><td>5.4</td><td>12.7</td><td>3.0</td><td>11.6 2.3</td><td>9.2</td><td>5.3</td><td>10.3</td><td>4.1</td><td>9.2</td><td>2.7</td><td>5.9</td></tr><tr><td>Featte 3DGS</td><td>10.6</td><td>28.6</td><td>31.5</td><td>54.7</td><td>35.0</td><td>64.6</td><td>27.3</td><td>45.9</td><td>40.1</td><td>64.5 31.0</td><td>71.2</td><td>41.4</td><td>70.2</td><td>27.4</td><td>49.4</td><td>26.9</td><td>588.3</td></tr><tr><td>aussian Grouping</td><td>42.9</td><td>69.7</td><td>19.4</td><td>46.2</td><td>23.1</td><td>47.4</td><td>40.6</td><td>81.5</td><td>50.0</td><td>88.5 39.2</td><td>86.2</td><td>55.3</td><td>85.5</td><td>36.6</td><td>65.3</td><td>27.9</td><td>68.2</td></tr><tr><td>SAD-Splat (Ours)</td><td>69.2</td><td>85.7</td><td>65.2</td><td>84.0</td><td>61.2</td><td>78.6</td><td>56.8</td><td>87.7</td><td>69.2</td><td>95.1 53.8</td><td>91.4</td><td>67.2</td><td>91.7</td><td>57.1</td><td>84.7</td><td>47.4</td><td>90.8</td></tr></table>

Table 2: Quantitative comparisons with state-of-the-art methods across various scene types are presented. Metrics, including mean Intersection over Union (mIoU) and mean Accuracy (mAcc), are reported in percentages (%). The highest scores are highlighted in bold.

<!-- image-->  
Figure 4: Visual comparison of segmentation results produced by different methods on the City and Country datasets.

2024), and Feature 3DGS (Zhou et al. 2024). All methods were evaluated on our proposed 3D-AS.

Quantitative results are presented in Table 2, with visualizations shown in Fig. 4. Although 2D methods exhibit competitive performance in certain scenes, their lack of explicit 3D structure modeling often leads to poor spatial consistency across multiple views. 3D open-vocabulary methods show unstable performance in complex scenes with semantic ambiguity. In contrast, SAD-Splat achieves the best overall performance across all types of scenes, demonstrating its superiority in balancing high segmentation accuracy and compact representation. Its capabilities in multi-view semantic integration and structure-aware sparse modeling make it an efficient and scalable solution for the 3D-AVS-SS task.

## Ablation Study

To evaluate the individual contributions of modules in SAD-Splat, a series of ablation experiments was designed. SAD-Splat comprises two primary components: the Gaussian Point Drop module and the Pseudo-Label Generation module. The Gaussian Point Drop module integrates learnable structural validity and semantic confidence-guided mechanisms to filter 3D Gaussian points during training dynamically, enabling adaptive compression of the model structure. The Pseudo-Label Generation module leverages the feature extraction and semantic matching capabilities of SAM and GeoRSCLIP to generate high-quality pseudo-labels for unlabeled images, thereby enhancing the coverage and diversity of supervision signals.

To validate the effectiveness of these components, the following comparative experimental setups were constructed under the City scene:

â¢ Net-1: Base model with 3 GT supervision (Baseline).

â¢ Net-2: Baseline with Gaussian Point Drop module $( P _ { \mathrm { b a s e } } = 1 . 0 )$

â¢ Net-3: Baseline with Pseudo-Label Generation module (generating pseudo-labels for the remaining 60 unlabeled images).

â¢ Net-4: Baseline with both Gaussian Point Drop and Pseudo-Label Generation modules (complete SAD-

<table><tr><td>Network</td><td>mIoU (%)</td><td>Accuracy (%)</td></tr><tr><td>Net-1</td><td>64.44</td><td>82.01</td></tr><tr><td>Net-2</td><td>64.95</td><td>82.68</td></tr><tr><td>Net-3</td><td>64.68</td><td>82.10</td></tr><tr><td>Net-4</td><td>65.22</td><td>82.74</td></tr></table>

Table 3: Performance comparison across various network configurations is presented. The top-performing results are highlighted in bold.

<table><tr><td>Network</td><td>mIoU (%)</td><td>Accuracy (%)</td></tr><tr><td>SAD-Splat</td><td>65.22</td><td>82.74</td></tr><tr><td>Semantic Confidence Drop</td><td>63.79</td><td>82.57</td></tr><tr><td>Learnable Structure Drop</td><td>64.01</td><td>82.70</td></tr></table>

Table 4: Impact of Various Drop Strategies on Experimental Performance. Best results are highlighted in bold.

Splat).

Table 3 presents the performance comparison of each configuration. Here, âthe base modelâ refers to a 3D semantic segmentation framework built upon Feature 3DGS and trained with ground-truth segmentation labels from three viewpoints (3 GT supervision). Net-1 serves as the baseline, achieving an mIoU of 64.44% and an accuracy of 82.01%. Net-2 incorporates the Gaussian Point Drop module, guided by semantic confidence and learnable structural validity, resulting in improved performance with an mIoU of 64.95% and an accuracy of 82.68%. This demonstrates that selectively removing redundant or inefficient Gaussian points enhances representational capacity. Net-3 integrates the Pseudo-Label Generation module, which provides semantic pseudo-supervision for unlabeled images, yielding an mIoU of 64.68% and an accuracy of 82.10%. This highlights the effectiveness of additional supervision signals in improving model generalization. Net-4 combines both modules, achieving the best performance with an mIoU of 65.22% and an accuracy of 82.74%, confirming the synergistic effect of the modules in SAD-Splat for enhancing 3D semantic understanding.

## Analysis of the Contribution of Gaussian Point Drop submodules

To assess the contributions of the two submodules in SAD-Splatâs Gaussian Point Drop mechanism, we conducted targeted ablation experiments. Table 4 presents configurations labeled âSemantic Confidence Dropâ and âLearnable Structure Drop,â using only semantic confidence-based or learnable structural drop, respectively. SAD-Splat integrates both mechanisms.

Results show that Learnable Structure Drop achieves an mIoU of 64.01%, while Semantic Confidence Drop yields an mIoU of 63.79%. The former uses $L _ { 0 }$ regularization and Hard Concrete distribution to reduce redundant points, but may discard critical information. The latter relies on a semantic confidence map from SAM and GeoRSCLIP to assess semantic reliability, yet may overly eliminate valuable low-confidence points. In contrast, SAD-Splat, which combines both, achieves an mIoU of 65.22% and a precision of 82.74%, demonstrating the complementarity of the submodules: structural drop enhances sparsity, while semantic drop targets ambiguous regions, significantly improving performance and validating the Gaussian Point Drop design. This synergistic mechanism ensures model robustness in complex scenes, enhancing its potential for 3D-AVS-SS tasks.

<table><tr><td>Network</td><td>mIoU (%)</td><td>Accuracy (%)</td><td>Gaussian Points</td></tr><tr><td>baseline</td><td>64.44</td><td>82.01</td><td>919,553</td></tr><tr><td> $P _ { \mathrm { b a s e } } { = } 0 . 5$ </td><td>64.66</td><td>82.71</td><td>315,218</td></tr><tr><td> $P _ { \mathrm { b a s e } } { = } 1 . 0$ </td><td>65.22</td><td>82.74</td><td>162,015</td></tr><tr><td> $P _ { \mathrm { b a s e } } { = } 1 . 5$ </td><td>64.74</td><td>82.55</td><td>87,547</td></tr><tr><td> $P _ { \mathrm { b a s e } } { = } 2 . 0$ </td><td>63.83</td><td>82.38</td><td>48,960</td></tr></table>

Table 5: Influence of the Base Drop Rate $P _ { b a s e }$ on Model Performance. Best results are highlighted in bold.

## Parameter Analysis

## Impact of Base Drop Rate $\ P _ { \mathbf { b a s e } }$ on Model Performance

We conducted a parameter analysis to investigate the impact of the base drop rate $P _ { \mathrm { b a s e } }$ on the performance and model sparsity of SAD-Splat. As shown in Table $^ { 5 , }$ as $P _ { \mathrm { b a s e } }$ increases, the number of Gaussian points gradually decreases, thereby reducing the model size (from 315,218 points at $P _ { \mathrm { b a s e } } = 0 . 5 \mathrm { t o } 4 8 , 7 6 0$ points at $P _ { \mathrm { b a s e } } = 2 . 0 )$

When $P _ { \mathrm { b a s e } } ~ = ~ 1 . 0$ , the model achieves optimal performance, attaining the highest mIoU (65.22%) and accuracy (82.74%) with only 162,015 Gaussian points, significantly fewer than the baselineâs 919,553 points. At $P _ { \mathrm { b a s e } } = 0 . 5 .$ , the model substantially reduces the number of Gaussian points compared to the baseline, while achieving a modest improvement in accuracy (82.71%) and a slight increase in mIoU (64.66%). When $P _ { \mathrm { b a s e } } ~ \geq ~ 1 . 5$ , segmentation performance noticeably declines, indicating that an excessive drop impairs performance. These results suggest that $P _ { \mathrm { b a s e } } = 1 . 0$ strikes the best balance between segmentation accuracy and model compactness.

## Conclusion

In this paper, we propose SAD-Splat, a novel approach for 3D-AVS-SS that addresses the challenges of semantic ambiguity and structural redundancy. We introduce a Gaussian point drop module, which integrates semantic confidence estimation with a learnable sparsity mechanism based on the Hard Concrete distribution. Additionally, we design a highconfidence pseudo-label generation pipeline that effectively leverages 2D foundation models to enhance supervision under limited ground-truth labels. To support this task, we construct a challenging dataset, 3D-AS, encompassing diverse and complex real-world aerial scenes. Experimental results demonstrate that SAD-Splat significantly reduces the number of Gaussian points in the final model while improving segmentation performance, achieving a compact yet expressive 3D representation. Our approach offers a promising direction for efficient and scalable 3D semantic understanding in remote sensing applications.

## References

Caron, M.; Touvron, H.; Misra, I.; Jegou, H.; Mairal, J.; Â´ Bojanowski, P.; and Joulin, A. 2021. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, 9650â9660.

Cen, J.; Fang, J.; Yang, C.; Xie, L.; Zhang, X.; Shen, W.; and Tian, Q. 2025. Segment any 3d gaussians. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 1971â1979.

Cherti, M.; Beaumont, R.; Wightman, R.; Wortsman, M.; Ilharco, G.; Gordon, C.; Schuhmann, C.; Schmidt, L.; and Jitsev, J. 2023. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2818â2829.

Dong, X.; Bao, J.; Zheng, Y.; Zhang, T.; Chen, D.; Yang, H.; Zeng, M.; Zhang, W.; Yuan, L.; Chen, D.; et al. 2023. Maskclip: Masked self-distillation advances contrastive language-image pretraining. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10995â11005.

Fu, S.; Hamilton, M.; Brandt, L.; Feldman, A.; Zhang, Z.; and Freeman, W. T. 2024. Featup: A model-agnostic framework for features at any resolution. arXiv preprint arXiv:2403.10516.

Huang, L.; Jiang, B.; Lv, S.; Liu, Y.; and Fu, Y. 2023. Deeplearning-based semantic segmentation of remote sensing images: A survey. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 17: 8370â8396.

Kerbl, B.; Kopanas, G.; Leimkuhler, T.; and Drettakis, G. Â¨ 2023. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4): 139â1.

Kerr, J.; Kim, C. M.; Goldberg, K.; Kanazawa, A.; and Tancik, M. 2023. Lerf: Language embedded radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, 19729â19739.

Kirillov, A.; Mintun, E.; Ravi, N.; Mao, H.; Rolland, C.; Gustafson, L.; Xiao, T.; Whitehead, S.; Berg, A. C.; Lo, W.-Y.; et al. 2023. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, 4015â4026.

Li, B.; Weinberger, K. Q.; Belongie, S.; Koltun, V.; and Ranftl, R. 2022. Language-driven semantic segmentation. arXiv preprint arXiv:2201.03546.

Liu, K.; Zhan, F.; Zhang, J.; Xu, M.; Yu, Y.; El Saddik, A.; Theobalt, C.; Xing, E.; and Lu, S. 2023. Weakly supervised 3d open-vocabulary segmentation. Advances in Neural Information Processing Systems, 36: 53433â53456.

Louizos, C.; Welling, M.; and Kingma, D. P. 2017. Learning sparse neural networks through L 0 regularization. arXiv preprint arXiv:1712.01312.

Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.; Ramamoorthi, R.; and Ng, R. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1): 99â106.

Piekenbrinck, J.; Schmidt, C.; Hermans, A.; Vaskevicius, N.; Linder, T.; and Leibe, B. 2025. OpenSplat3D: Open-Vocabulary 3D Instance Segmentation using Gaussian Splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, 5246â5255.

Qin, M.; Li, W.; Zhou, J.; Wang, H.; and Pfister, H. 2024. Langsplat: 3d language gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 20051â20060.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, 8748â8763. PmLR.

Rahnemoonfar, M.; Chowdhury, T.; and Murphy, R. 2022. RescueNet: A high resolution UAV semantic segmentation benchmark dataset for natural disaster damage assessment. arXiv preprint arXiv:2202.12361.

Ravi, N.; Gabeur, V.; Hu, Y.-T.; Hu, R.; Ryali, C.; Ma, T.; Khedr, H.; Radle, R.; Rolland, C.; Gustafson, L.; et al. 2024.Â¨ Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714.

Zhang, Z.; Zhao, T.; Guo, Y.; and Yin, J. 2024. RS5M and GeoRSCLIP: A large scale vision-language dataset and a large vision-language model for remote sensing. IEEE Transactions on Geoscience and Remote Sensing.

Zhou, S.; Chang, H.; Jiang, S.; Fan, Z.; Zhu, Z.; Xu, D.; Chari, P.; You, S.; Wang, Z.; and Kadambi, A. 2024. Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21676â21685.

## Reproduction Checklist

## Methodology Description

â¢ Includes a conceptual outline and/or pseudocode description of AI methods introduced. (yes)

â¢ Clearly delineates statements that are opinions, hypothesis, and speculation from objective facts and results. (yes )

â¢ Provides well marked pedagogical references for lessfamiliar readers to gain background necessary to replicate the paper. (yes)

## Theoretical Contributions

â¢ Does this paper make theoretical contributions? (yes) If yes, please complete the list below.

â All assumptions and restrictions are stated clearly and formally. (yes)

â All novel claims are stated formally (e.g., in theorem statements). (yes)

â Proofs of all novel claims are included. (yes)

â Proof sketches or intuitions are given for complex and/or novel results. (yes)

â Appropriate citations to theoretical tools used are given. (yes)

â All theoretical claims are demonstrated empirically to hold. (NA)

â All experimental code used to eliminate or disprove claims is included. (NA)

## Datasets

â¢ Does this paper rely on one or more datasets? (yes) If yes, please complete the list below.

â A motivation is given for why the experiments are conducted on the selected datasets. (yes)

â All novel datasets introduced in this paper are included in a data appendix. (yes)

â All novel datasets introduced in this paper will be made publicly available upon publication of the paper with a license that allows free usage for research purposes. (yes)

â All datasets drawn from the existing literature are accompanied by appropriate citations. (NA)

â All datasets drawn from the existing literature are publicly available. (NA)

â All datasets that are not publicly available are described in detail, with explanation why publicly available alternatives are not scientifically satisficing. (NA)

## Computational Experiments

â¢ Does this paper include computational experiments? (yes / no)

If yes, please complete the list below.

â This paper states the number and range of values tried per (hyper-)parameter during development, along with the criterion used for selecting the final parameter setting. (partial)

â Any code required for pre-processing data is included in the appendix. (partial)

â All source code required for conducting and analyzing the experiments is included in a code appendix. (yes)

â All source code required for conducting and analyzing the experiments will be made publicly available upon publication of the paper with a license that allows free usage for research purposes. (yes)

â All source code implementing new methods have comments detailing the implementation, with references to the paper where each step comes from. (yes)

â If an algorithm depends on randomness, then the method used for setting seeds is described in a way sufficient to allow replication of results. (yes)

â This paper specifies the computing infrastructure used for running experiments (hardware and software). (yes)

â This paper formally describes evaluation metrics used and explains the motivation for choosing these metrics. (yes)

â This paper states the number of algorithm runs used to compute each reported result. (yes)

â Analysis of experiments goes beyond singledimensional summaries of performance to include measures of variation, confidence, or other distributional information. (no)

â The significance of any improvement or decrease in performance is judged using appropriate statistical tests (e.g., Wilcoxon signed-rank). (no)

â This paper lists all final (hyper-)parameters used for each model/algorithm in the paperâs experiments. (yes)