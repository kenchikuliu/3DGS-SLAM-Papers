# Reconstruction Using the Invisible: Intuition from NIR and Metadata for Enhanced 3D Gaussian Splatting

Gyusam Chang1, 2\* Tuan-Anh Vu2 Vivek Alumootil2

Harris Song2 Deanna Pham2 Sangpil Kim1â  M. Khalid Jawed2â 

1Korea University 2University of California, Los Angeles

{gsjang95, spk7}@korea.ac.kr {tuananh.vu, vivekalumootil, songharris2006, deannapham2004, khalidjm}@ucla.edu

## Abstract

While 3D Gaussian Splatting (3DGS) has rapidly advanced, its application in agriculture remains underexplored. Agricultural scenes present unique challenges for 3D reconstruction methods, particularly due to uneven illumination, occlusions, and a limited field of view. To address these limitations, we introduce NIRPlant, a novel multimodal dataset encompassing Near-Infrared (NIR) imagery, RGB imagery, textual metadata, Depth, and LiDAR data collected under varied indoor and outdoor lighting conditions. By integrating NIR data, our approach enhances robustness and provides crucial botanical insights that extend beyond the visible spectrum. Additionally, we leverage text-based metadata derived from vegetation indices, such as NDVI, NDWI, and the chlorophyll index, which significantly enriches the contextual understanding of complex agricultural environments. To fully exploit these modalities, we propose NIRSplat, an effective multimodal Gaussian splatting architecture employing a crossattention mechanism combined with 3D point-based positional encoding, providing robust geometric priors. Comprehensive experiments demonstrate that NIRSplat outperforms existing landmark methods, including 3DGS, CoR-GS, and InstantSplat, highlighting its effectiveness in challenging agricultural scenarios. The code and dataset are publicly available at: https://github.com/StructuresComp/3D-Reconstruction-NIR

## Introduction

3D reconstruction has become increasingly crucial across various fields, including robotics, autonomous driving, augmented reality, and agricultural monitoring. Traditional methods for reconstructing three-dimensional structures from twodimensional images often struggle to capture fine details, handle complex scenes, and maintain robustness under challenging environmental conditions. Recently, 3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) has emerged as a significant advancement, enabling smoother, more detailed, and computationally efficient reconstructions. Unlike traditional approaches that rely heavily on discrete point representations (Sinha et al. 2017; Li et al. 2018; Lin, Kong, and Lucey 2018; Nguyen et al. 2019), 3DGS represents each 3D point as a Gaussian distribution, effectively capturing uncertainties and spatial continuity in complex environments.

<!-- image-->  
Figure 1: Qualitative comparisons in a 3-view setup: rendered images from the ground truth, InstantSplat, and our method. We highlight our improved semantic understanding, particularly in regions of interest within the image, where our method more accurately captures meaningful structures and distinctions under novel lighting conditions. Zoom in for better visualization.

Despite its demonstrated success in general scenarios, 3DGS faces significant challenges when applied to agriculture. As shown in Fig. 1, these environments pose unique challenges, including unpredictable lighting variations (e.g., intense sunlight, low visibility, and sunset conditions), limited viewing angles, environmental instability due to weather fluctuations, and frequent occlusions by foliage. These factors can substantially degrade the performance of typical 3D reconstruction methods (Fan et al. 2024b; Zhang et al. 2024), resulting in incomplete or inaccurate plant modeling.

To overcome these limitations, we introduce the NIR-Plant dataset, specifically designed to address the unique challenges of agricultural environments. NIRPlant incorporates comprehensive multimodal data, including RGB images, Near-Infrared (NIR) imagery, and rich textual metadata. The dataset (see Tab. 1) comprises diverse indoor and outdoor lighting scenarios captured from multiple perspectives, including artificial illumination, direct sunlight, and sunset conditions. Integrating NIR imagery is particularly advantageous because NIR can capture plant-specific reflectance characteristics invisible to conventional RGB cameras, thus providing essential botanical information about plant health, water content, and structural integrity. For instance, high values of NDVI (Normalized Difference Vegetation Index) typically indicate robust vegetation health, NDWI (Normalized Difference Water Index) reflects water content, and chlorophyll index values directly correlate with photosynthetic efficiency and plant vigor. Such indices enrich our dataset and significantly enhance the modelâs ability to accurately interpret complex botanical scenarios, as shown in Fig. 1.

Moreover, textual metadata derived from both RGB and NIR images includes environmental conditions, precise lighting descriptions, and quantitative botanical indices, thus providing rich context for reconstructing detailed 3D models. Fusing this metadata with visual modalities enables our method to interpret plants more effectively and model them under diverse and challenging photometric conditions.

To leverage the full potential of our multimodal dataset, we propose NIRSplat, a novel Gaussian splatting framework optimized for multimodal data integration. NIRSplat employs a novel cross-attention mechanism (Zhu et al. 2020; Vaswani et al. 2017) that effectively combines NIR embeddings with RGB features. Our approach achieves superior scene understanding by exploiting the complementary strengths of RGB imagery and NIR-derived features. Moreover, inspired by the success of Vision-Language Models (VLMs) (Radford et al. 2021; Li et al. 2022a, 2023a; Liu et al. 2023a), we integrate textual embeddings derived from metadata descriptions to further enhance semantic understanding. This multimodal interaction is further enhanced by employing a novel 3D point-based positional encoding method, which leverages spatial coherence from geometric priors to align and enrich 2D image features with 3D spatial information.

We conducted extensive evaluations to validate our proposed method, comparing NIRSplat with state-of-the-art approaches. Our results show that NIRSplat outperforms existing methods in terms of reconstruction accuracy, robustness to varying environmental conditions, and visual quality. We provide detailed analyses that highlight the contributions of each modality to the overall improvement in performance. In summary, our key contributions include:

â¢ The introduction of NIRPlant, a comprehensive multimodal agricultural dataset integrating RGB, NIR, and detailed textual metadata, enabling robust 3D reconstruction under varying lighting and environmental conditions.

â¢ Development of NIRSplat, a multimodal Gaussian splatting framework employing cross-attention mechanisms and geometric priors, significantly improving scene reconstruction robustness.

â¢ Extensive comparative analyses demonstrate our approachâs effectiveness and advantages over leading methods, including 3DGS, CoR-GS, and InstantSplat, under diverse agricultural conditions (e.g., intense sunlight, occlusion, low visibility).

Table 1: Comparison with existing landmark dataset. R, D, N, S, L, and T denote RGB, Depth, NIR, Structured-Light Scanner (SLS), LiDAR, and Text, respectively. Lighting indicates whether there are various lighting conditions for supervision.
<table><tr><td>Dataset</td><td>Modality</td><td>Lighting</td><td># Scenes</td><td># Views</td><td>Metadata</td></tr><tr><td>Barron et al. (2022)</td><td>R</td><td>X</td><td>9</td><td>100-330</td><td>X</td></tr><tr><td>Knapitsch et al. (2017)</td><td>R</td><td>X</td><td>14</td><td>4-17</td><td>X</td></tr><tr><td>Toschi et al. (2023)</td><td>R</td><td>â</td><td>20</td><td>2000</td><td>X</td></tr><tr><td>Voynov et al. (2023)</td><td>R,D,S</td><td>X</td><td>107</td><td>100</td><td>X</td></tr><tr><td>NIRPlant (Ours)</td><td>R,D,N,L,T</td><td>â</td><td>34</td><td>360</td><td>â</td></tr></table>

## Related Works

## Method for 3D Reconstruction

3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) uses a set of 3D Gaussian parameters and differentiable splatting to represent and render scenes more efficiently than traditional radiance fields (Martin-Brualla et al. 2021; Garbin et al. 2021; Barron et al. 2021). Mip-Splat (Yu et al. 2024) is another scene rendering algorithm that constrains the size of 3D Gaussian primitives and mitigates aliasing and dilation issues present in 3DGS. CF-3DGS (Fu et al. 2024) reduces the burden of pre-computation by leveraging the temporal continuity from video and the explicit point cloud representation. CoR-GS (Zhang et al. 2024) identifies and suppresses inaccurate reconstruction using Co-pruning, considers Gaussians, and Pseudo-view co-regularization. Furthermore, another InstantSplat (Fan et al. 2024b) is compatible with both the above methods and specializes in low-image-count representations through a neural network representation similar to that of NSR, utilizing a Gaussian Bundle Adjustment (GauBA). SplatFields (Mihajlovic et al. 2024) designs and regularizes splat features as the outputs of a corresponding implicit neural field. Recently, CATSplat (Roh et al. 2024) introduced a generalizable transformer-based framework, addressing the inherent constraints in monocular settings.

## Dataset for 3D Reconstruction

Various 3D Reconstruction datasets have focused on advancements in lighting recognition and multimodal approaches. Mip-NeRF360 (Barron et al. 2022) synthesizes realistic object views in the real world, while MVimgNet enhances 3D capture through video-based 3D-aware signals (Yu et al. 2023). Adding on, ReLight and Tanks are designed to address lighting variation with different materials (Toschi et al. 2023; Voynov et al. 2023; Knapitsch et al. 2017). Previous adaptations of NeRF to real-world environments through LEGO bricks (Li et al. 2023b) and famous city sites (Martin-Brualla et al. 2021) improve lighting capture by training from photo datasets. OmniObject3D addresses surface reconstruction for dense and sparse-view surfaces (Wu et al. 2023). Additionally, GauU-Scene (Xiong, Li, and Li 2024) supports large-scale scene reconstruction using Gaussian Splatting for real-time scanning. NeRFBK (Yan et al. 2023) utilizes both real and synthetic data to capture objects of varying materials and lighting conditions, thereby comparing NeRF in outdoor views and for transparent objects. UniSDF (Wang et al. 2024a) is another dataset that utilizes NeRF to capture 3D scenes with reflections, combining the traditional SDF with radiance fields to render scenes with and without reflections.

<!-- image-->  
Figure 2: (a) Data Acquisition Platform: (Top) Sensor installation, and (Bottom) Sensor configuration. (b) Sensor configuration and data collection procedure for both indoor and outdoor settings.

## Our Multimodal NIRPlant Dataset

Collecting comprehensive multimodal datasets in diverse environmental conditions is inherently challenging, even within controlled laboratory settings. The manual process involved in collecting, observing, and annotating multiple data types, such as RGB and NIR imagery, and metadata, is labor-intensive, time-consuming, and costly. Moreover, achieving diversity in visual data and ensuring high-quality annotations significantly complicates the process. Please refer to the supplementary material for additional details.

## Data Acquisition Platform

Our primary objective is to develop a comprehensive and versatile 3D plant reconstruction dataset under diverse realworld lighting conditions. However, dynamic environmental factors such as wind, shadows, and fluctuating sunlight pose significant challenges to data reliability. To mitigate these effects, we designed a controlled lighting configuration (see Fig. 2) that captures data during critical illumination periods, allowing us to observe lighting variance systematically. Precisely, objects were positioned to capture precise lighting conditions at defined times of the day (i.e., noon and sunset). Furthermore, we utilized a multimodal sensor setup consisting of a ZED 2i and Nikon D3400 HD RGB cameras, an Alvium 1800 U-501 NIR sensor, and a Neuvition Titan S2-70C LiDAR sensor, ensuring accurate alignment and consistent distance between the objects and sensors. To enhance data quality under natural conditions, automatic adjustments for focus, exposure, and gain were employed to maintain consistency and adaptivity in capturing agricultural data, thus enabling accurate calibration and precise extraction of camera perspectives essential for reliable 3D reconstruction.

## Data Construction and Processing

Collecting precise camera poses in agricultural environments is inherently challenging due to the dynamic nature of plants, whose structures change rapidly in response to environmental factors. To overcome this issue, extensive data were collected indoors and outdoors under strictly consistent conditions. Specifically, each object was captured from 360 multi-modal data samples per scene, and ground truth models were constructed using the landmark Structure-from-Motion (SfM)

<!-- image-->  
Figure 3: Hierarchical organization of the dataset taxonomy.

technique (Schonberger and Frahm 2016), ensuring high ac- Â¨ curacy and robustness, particularly for texture-rich environments. Additionally, precise reconstruction was prioritized by meticulously removing background information from object images. In total, our dataset contains comprehensive multimodal data (as described in Tab. 1), covering four distinct lighting scenarios with 360 viewpoints per scene across up to 10 different plant categories, ensuring broad diversity and representativeness (see Fig. 3).

## Dataset Specifications and Statistics

Data Organization. As illustrated in Fig. 3, our NIRPlant dataset categorizes plant data under four primary lighting conditions: artificial light, strong sunlight, low light, and normal daylight. It encompasses up to 10 diverse plant species, further classified into three distinct size categories (small, medium, and large). Each plant category and lighting condition was captured consistently from 360 viewpoints to ensure robust perspective coverage. This systematic data acquisition process was uniformly applied across RGB, NIR, Depth, and LiDAR. Note that botanic-aware prompts are generated for each scene, as illustrated in the supplementary material. Additionally, to effectively extract discriminative NIR signals, comparisons were conducted against artificial plants (Art 1 and Art 2). Leveraging this comprehensive multimodal dataset structure, we aim to enhance the understanding of plant-specific characteristics under various environments, thus substantially improving the performance and robustness of 3D reconstruction methods.

Dataset Split. Considering the practical agricultural environment, we adopted a sparse-view approach inspired by InstantSplat (Fan et al. 2024a). Specifically, from each set of 24 RGB, NIR, and textual metadata viewpoints, we randomly sampled 3, 6, and 12 views for training and testing purposes, respectively. This sampling strategy simulates realworld scenarios where limited views are common, ensuring the datasetâs applicability and the generalization of reconstruction algorithms in realistic agricultural contexts.

<!-- image-->  
Figure 4: The overall architecture of NIRSplat framework. NIRSplat efficiently processes tri-modal inputs consisting of NIR, RGB, and Text, enabling joint reasoning. The details of the prompt engineering are included in the supplementary document.

## Our Proposed Method

In this section, we describe our multimodal method in detail, emphasizing the multimodal initialization through 3D positional encoding, the Transformer-based interactions for modality fusion, and the multimodal loss and regularization mechanisms. Note that the technical background necessary for understanding our proposed method is provided in the supplementary material.

<table><tr><td>Symbol</td><td>Description</td></tr><tr><td> $\mathbf { G }$   $\{ \mu , \alpha , \Sigma , c \}$   $\mathbf { P } = \{ \mathbf { p } _ { i } \} _ { i = 1 } ^ { N }$   $\mathbf { T } = \left[ R \mid \mathbf { t } \right]$   $\mathbf { K }$ </td><td>Set of learnable Gaussian primitives Gaussian representations 3D point maps Camera pose Camera intrinsic matrix</td></tr><tr><td> $\mathbf { u } _ { i }$   $\lambda$ </td><td>2D projection of point  $\mathbf { p } _ { i }$  via T and K Learnable frequency scale</td></tr><tr><td> $\operatorname { P E } _ { i }$   $\mathrm { F _ { r g b } , F _ { n i r } , F _ { t x t } }$ </td><td>Positional encoding from projected 2D point Feature representations extracted from</td></tr><tr><td></td><td>RGB images, NIR images, and text in- puts, respectively</td></tr><tr><td> $\mathrm { { F _ { n r } } }$   $\mathrm { F _ { n t r } }$ </td><td>Joint visual representation obtained by fusing  $\mathrm { F _ { r g b } }$  and  $\mathrm { F _ { \mathrm { n i r } } }$ </td></tr><tr><td></td><td>Multimodal feature representation ob- tained by integrating  $\mathrm { F _ { n r } }$  with  $\mathrm { F _ { t x t } }$ </td></tr></table>

## Gaussian-guided Positional Anchoring

Recently, the explicit way (Godard, Mac Aodha, and Brostow 2017; Godard et al. 2019; Yang et al. 2024) to interact with the 3D priors is to estimate depths from input RGB images. However, such an approach profoundly limits the advantage of 3DGS (i.e., real-time NVS) by demanding additional deeplearning capacity. To mitigate this, we propose a lightweight and efficient alternative: Gaussian-guided Positional Anchoring inspired by (Shu et al. 2023; Liu et al. 2022), which provides strong geometric clues from initialized Gaussian positions without requiring external depth supervision.

Positional Anchoring leveraging Gaussian means. We leverage MASt3R (Leroy, Cabon, and Revaud 2024a) to predict an initial dense 3D point map $\{ { \bf p } _ { i } \} _ { i = 1 } ^ { N }$ 1, which serves as the initialization for our Gaussian representation ${ \textbf { G } } =$ $\{ \mu _ { i } , \Sigma _ { i } , \alpha _ { i } , c _ { i } \}$ , where $\mu _ { i } = \mathbf { p } _ { i }$ i denotes the Gaussian center. Simultaneously, a coarse camera extrinsic matrix $\mathrm { T } = \left\lceil R \right\rceil$ $\mathbf { t } ] \in \mathbf { S E } ( 3 )$ is obtained per view, also from MASt3R. Given the current estimate of camera pose T and the intrinsic matrix $\mathbf { K } \in \mathbb { R } ^ { 3 \times 3 }$ , we project each 3D point $\mathbf { p } _ { i } \in \mathbb { R } ^ { 3 }$ onto the 2D image plane as below:

$$
\tilde { \mathbf { u } } _ { i } = \mathbf { K } \cdot ( R \cdot \mathbf { p } _ { i } + \mathbf { t } ) \in \mathbb { R } ^ { 3 } ,\tag{1}
$$

$$
\mathbf { u } _ { i } = \left[ \frac { \tilde { u } _ { i } ^ { x } } { \tilde { u } _ { i } ^ { z } } , \frac { \tilde { u } _ { i } ^ { y } } { \tilde { u } _ { i } ^ { z } } \right] \in \mathbb { R } ^ { 2 } .\tag{2}
$$

Each projected 2D location $\mathbf { u } _ { i }$ serves as a spatial anchor, from which we derive a positional embedding using either sinusoidal encoding or a lightweight MLP.

$$
\mathrm { P E } _ { i } = \Phi \left( \left[ \sin ( \lambda ^ { \top } \mathbf { u } _ { i } ) \oplus \cos ( \lambda ^ { \top } \mathbf { u } _ { i } ) \right] \right) ,\tag{3}
$$

where Î» represents a learnable frequency scale and â denotes concatenation. Î¦(Â·) is a multilayer perceptron (MLP) that maps the encoded coordinates to a latent embedding space. This formulation enables efficient and geometry-aware interaction with 3D points directly on the image plane by providing a unified positional reference. Crucially, it preserves spatial correspondence and depth continuity without incurring the cost of full-scale depth estimation, thus maintaining the efficiency and lightweight design of 3DGS.

## NIRSplat: A Multimodal Gaussian Splatting

Bridging the invisible: NIR-RGB Coupling. 3DGS (Kerbl et al. 2023) introduces a powerful Gaussian-based representation that enables real-time novel view synthesis, driving substantial progress across numerous 3D vision applications. However, in outdoor agricultural scenarios, where sensor configurations are often sparse and viewpoint coverage is inherently limited, we observe a considerable performance drop due to inconsistent lighting, occlusion, and textureless surfaces. To overcome these challenges, we incorporate nearinfrared (NIR) sensing as a complementary modality to RGB. NIR images capture electromagnetic wavelengths beyond the visible spectrum, revealing latent structural information such as chlorophyll absorption, leaf water content, and surface reflectance properties that are often invisible in RGB. By leveraging this spectral prior, we aim to enhance feature robustness under adverse imaging conditions. To this end, we design a Transformer-based NIR-RGB fusion module using a deformable cross-attention mechanism D Attn (Vaswani et al. 2017; Zhu et al. 2020). Let $\mathbf { F } _ { r g b } = \{ f _ { r g b } ^ { i } \} _ { i = 1 } ^ { N }$ and $\mathbf { F } _ { n i r } = \{ f _ { n i r } ^ { i } \} _ { i = 1 } ^ { N }$ be the extracted features from RGB and NIR branches. Each modality is augmented with a shared positional encoding $\mathrm { P E } _ { i } = \mathrm { P E } [ : , u _ { i } , v _ { i } ]$ , and fused via:

$$
\mathbf { F } _ { n r } ^ { ( i ) } = D \_ A t t n ( f _ { r g b } ^ { ( i ) } \oplus \mathbb { P } \mathrm { E } _ { i } , \ f _ { n i r } ^ { ( i ) } \oplus \mathbb { P } \mathrm { E } _ { i } , \ f _ { n i r } ^ { ( i ) } \oplus \mathbb { P } \mathrm { E } _ { i } )\tag{4}
$$

Here, D $. A t t n ( \cdot )$ applies a multi-head deformable attention operation. This formulation allows the RGB features to selectively attend to informative NIR signals guided by spatial anchors from the shared positional encoding. The fused representation is obtained by stacking L attention layers, yielding the final robust cross-modal feature set: $\mathbf { F } _ { n r } = \left\{ \mathbf { f } _ { n r } ^ { i } \right\} _ { i = 1 } ^ { N } ,$ where $\mathbf { F } _ { n r } ^ { ( i ) } \in \mathbb { R } ^ { C }$ . This transformer-driven NIR-RGB interaction enables effective exploitation of cross-spectral cues under limited views, leveraging both radiometric contrast from NIR and geometric alignment via positional encoding. As demonstrated in our experiments, this mechanism significantly enhances scene understanding and 3D reconstruction quality under real-world agricultural constraints.

Bridging the invisible: RGB-Text Coupling. Vision-Language Models (VLMs) (Radford et al. 2021; Alayrac et al. 2022; Li et al. 2022a,b; Zhang, Li, and Bing 2023; Li et al. 2023a; Liu et al. 2023a) have recently achieved striking success across a wide range of tasks by tightly coupling visual inputs with rich textual descriptions. Despite their proven potential, these models remain largely unexplored in the domain of agricultural 3D reconstruction (i.e., a field that urgently demands robust, high-level scene understanding to support smart farming systems). To address this gap, we propose a Transformer-based RGB-Text interaction module that semantically bridges RGB features with language-derived plant attributes, enabling better recognition of hard samples (e.g., small objects, fine structures, and hard-to-perceive regions). Primarily, we observe that various factors (e.g., descriptions, object attributes, environmental cues) in text prompts significantly contribute to view understanding (Oh et al. 2024; In Lee et al. 2024; Roh et al. 2024; Lee et al. 2024) by guiding the modelâs attention and perspectives. Inspired by this, we generate botanical-aware prompts $\mathcal { T } \in \mathbb { R } ^ { L \times C }$ that encapsulate detailed semantic information, such as vegetation index (e.g., NDVI, NDWI), structural traits (e.g., leaf shape, stem thickness), phenological stages (e.g., sprouting, flowering), and context (e.g., lighting, occlusion), as detailed in the supplementary document. These prompts are first encoded using a pre-trained VLM to obtain tokenlevel text features: $\mathbf { F } _ { t x t } = \{ f _ { t x t } ^ { ( 1 ) } , f _ { t x t } ^ { ( 2 ) } , . . . , f _ { t x t } ^ { ( L ) } \}$ , where each $f _ { t x t } ^ { ( i ) } \in \mathbb { R } ^ { C }$ represents a contextualized embedding. To align language and vision, we also leverage the deformable attention mechanism D Attn (Vaswani et al. 2017; Zhu et al. 2020), injecting shared positional priors via PE (see Eq. 3) in the same manner. We formulate the multimodal features from the NIR-RGB fusion $\mathbf { F } _ { n r } ^ { ( i ) }$ and textual tokens $f _ { t x t } ^ { ( i ) }$ at pixel coordinate $( u _ { i } , v _ { i } )$ as follows:

$$
\mathbf { F } _ { n t r } ^ { ( i ) } = D _ { - } A t t n ( f _ { n r } ^ { ( i ) } \oplus \mathbb { P } \mathrm { E } _ { i } , ~ f _ { t x t } ^ { ( i ) } \oplus \mathbb { P } \mathrm { E } _ { i } , ~ f _ { t x t } ^ { ( i ) } \oplus \mathbb { P } \mathrm { E } _ { i } )\tag{5}
$$

Here, â denotes vector concatenation, and the attention module facilitates fine-grained alignment between visual and linguistic features at both spatial and semantic levels. The resulting multimodal feature $\mathbf { F } _ { \mathrm { n t r } }$ integrates geometric anchors and contextual cues and is subsequently processed by a lightweight feed-forward decoder: $\{ \mu , { \bar { \alpha } } , { \bf { \bar { \alpha } } } , c \} \ =$ $\dot { \mathrm { M L P } } _ { \mathrm { g a u s s } } ( { \bf F } _ { n t r } )$ , where $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ denotes the 3D Gaussian mean, Î± is the opacity, $\pmb { \Sigma } \in \mathbb { R } ^ { 3 \times 3 }$ represents the anisotropic covariance (or its low-rank approximation), and c is the RGB appearance feature. These learned G are directly fed into a 3D Gaussian Splatting renderer equipped with an Adaptive Density Control (ADC) mechanism, allowing efficient, robust, and botanic-aware scene reconstruction under novel agricultural scenarios (i.e., insufficient visual clues).

## Cross-Modal Gaussian Field Reasoning

To successfully render a set of Gaussians G and corresponding pose T , we adopt a Gaussian rasterization as a differentiable operator following Gaussian Bundle Adjustment (Fan et al. 2024a) in a self-supervised manner. Specifically, after a highly informative cross-modal fusion phase, the refined pose T and Gaussian field G are jointly optimized by minimizing the photometric rendering loss:

$$
\mathbf { G } ^ { * } \mathbf { T } ^ { * } = \arg \operatorname* { m i n } _ { \mathbf { G } , \mathbf { T } } \sum _ { v \in N } \sum _ { i = 1 } ^ { H W } \Big \| \tilde { C } _ { v } ^ { i } ( \mathbf { G } , \mathrm { T } ) - C _ { v } ^ { i } ( \mathbf { G } , \mathrm { T } ) \Big \| ,\tag{6}
$$

where C and $\tilde { C }$ are the rasterization function and the observed 2D images, respectively. Consequently, it is worth noting that this formulation facilitates rapid optimization, seamlessly incorporating complementary multimodal knowledge into the underlying 3D Gaussian representation.

## Experiments

Baselines. We selected recent state-of-the-art methods for comparison, including 3DGS (Kerbl et al. 2023), Splat-Fields (Mihajlovic et al. 2024), InstantSplat (Fan et al. 2024b) and CoR-GS (Zhang et al. 2024). These methods efficiently leverage a Gaussian parameter in real time, optimizing a position Âµ, an opacity Î±, a covariance $\pmb { \Sigma } \in \mathbb { R } ^ { 3 \times 3 }$ , and spherical harmonics (color) c with trivial computational overhead. Additionally, we adopt pose-free methods, Nope-NeRF (Bian et al. 2023) and CF-3DGS (Fu et al. 2024), which are supported by monocular depth maps and ground-truth camera intrinsics, following InstantSplat. Please refer to the supplementary document for detailed experimental setups.

Table 2: Main Performance with SOTA techniques (Fan et al. 2024b; Zhang et al. 2024; Mihajlovic et al. 2024) on NIRPlant dataset. We conduct experiments with 3, 6, and 12 view setups and calculate traditional three metrics: SSIM, PSNR, and LPIPS. 200, 1k, 10k and 30k denote iterations. Note that bold values indicate the best performance. Gray shading indicates Ours.
<table><tr><td rowspan="2">Method</td><td colspan="3">SSIM (â)</td><td colspan="3">PSNR(â)</td><td colspan="3">LPIPS (â)</td></tr><tr><td>3-view</td><td>6-view</td><td>12-view</td><td>3-view</td><td>6-view</td><td>12-view</td><td>3-view</td><td>6-view</td><td>12-view</td></tr><tr><td>3DGS</td><td>0.5074</td><td>0.5590</td><td>0.6531</td><td>14.1552</td><td>15.5586</td><td>17.4352</td><td>0.4586</td><td>0.4469</td><td>0.4033</td></tr><tr><td>CoR-GS-1k</td><td>0.7179</td><td>0.7642</td><td>0.8081</td><td>16.3494</td><td>17.1991</td><td>19.5895</td><td>0.4124</td><td>0.3191</td><td>0.2289</td></tr><tr><td>CoR-GS-10k</td><td>0.7285</td><td>0.7776</td><td>0.8287</td><td>16.7118</td><td>18.8023</td><td>20.8714</td><td>0.4049</td><td>0.3094</td><td>0.2281</td></tr><tr><td>CoR-GS-30k</td><td>0.7143</td><td>0.7611</td><td>0.8131</td><td>15.8925</td><td>17.5348</td><td>20.2927</td><td>0.4120</td><td>0.3405</td><td>0.2489</td></tr><tr><td>SplatFields-1k</td><td>0.7429</td><td>0.7647</td><td>0.7886</td><td>11.5490</td><td>13.6087</td><td>13.4497</td><td>0.4301</td><td>0.3787</td><td>0.3164</td></tr><tr><td>SplatFields-10k</td><td>0.7624</td><td>0.7799</td><td>0.8070</td><td>12.1196</td><td>14.6183</td><td>14.4463</td><td>0.3965</td><td>04017</td><td>0.2898</td></tr><tr><td>SplatFields-30k</td><td>0.7664</td><td>0.7802</td><td>0.8163</td><td>12.8751</td><td>14.1314</td><td>15.6037</td><td>0.3764</td><td>0.3754</td><td>0.2721</td></tr><tr><td>InstantSplat-200</td><td>0.7559</td><td>0.7604</td><td>0.7720</td><td>17.6177</td><td>17.9250</td><td>18.5293</td><td>0.3048</td><td>0.2943</td><td>0.2784</td></tr><tr><td>InstantSplat-1k</td><td>0.7984</td><td>0.8126</td><td>0.8134</td><td>18.3849</td><td>18.9233</td><td>19.033</td><td>0.2797</td><td>0.2689</td><td>0.2438</td></tr><tr><td>NIRSplat-200</td><td>0.7906</td><td>0.8099</td><td>0.8174</td><td>18.1747</td><td>18.7103</td><td>19.1921</td><td>0.2371</td><td>0.2267</td><td>0.2229</td></tr><tr><td>NIRSplat-1k</td><td>0.8268</td><td>0.8311</td><td>0.8421</td><td>20.7182</td><td>21.0169</td><td>2.0814</td><td>0.2070</td><td>0.2071</td><td>0.2080</td></tr></table>

Table 3: Ablation study on various configurations.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Configuration</td><td colspan="3">SSIM(â)</td><td colspan="3">PSNR (â)</td><td colspan="3">LPIPS (â)</td></tr><tr><td>3-view</td><td>6-view</td><td>12-view</td><td>3-view</td><td>6-view</td><td>12-view</td><td>3-view</td><td>6-view</td><td>12-view</td></tr><tr><td rowspan="7">InstantSplat-S</td><td> $I _ { r g b } \ \mathrm { { o n l y } }$ </td><td>0.7984</td><td>0.8126</td><td>0.8134</td><td>18.3849</td><td>18.9233</td><td>19.0333</td><td>0.2797</td><td>0.2689</td><td>0.2438</td></tr><tr><td> $\bar { I _ { r g b } } , \bar { I _ { n i r } }$ </td><td>0.7096</td><td>0.7383</td><td>0.7426</td><td>169913</td><td>17.6154</td><td>17.6345</td><td>0.2431</td><td>0.2265</td><td>0.2264</td></tr><tr><td> $\bar { F _ { r g b } } \oplus F _ { n i r }$ </td><td>0.8049</td><td>0.8079</td><td>0.8128</td><td>18.0318</td><td>18.7785</td><td>19.0927</td><td>0.3091</td><td>0.2923</td><td>0.2881</td></tr><tr><td> $\tilde { F _ { r g b } } \oplus F _ { n i r } \oplus F _ { t x t }$ </td><td>0.7875</td><td>0.7883</td><td>0.7938</td><td>16.4174</td><td>17.1859</td><td>17.5160</td><td>0.2933</td><td>0.2742</td><td>0.2634</td></tr><tr><td> $\bar { F _ { r g b } } + F _ { n i }$ </td><td>0.7605</td><td>0.7747</td><td>0.7789</td><td>16.1966</td><td>16.9488</td><td>17.1367</td><td>0.3623</td><td>0.3503</td><td>0.3505</td></tr><tr><td>c  $F _ { r g b } + F _ { n i r } + F _ { t x t }$ </td><td>0.7514</td><td>0.7671</td><td>0.7735</td><td>14.6166</td><td>15.4207</td><td>16.5871</td><td>0.3405</td><td>0.3353</td><td>0.3246</td></tr><tr><td></td><td></td><td>0.8139</td><td>0.8160</td><td>18.8696</td><td>18.7132</td><td>19.1866</td><td>0.2765</td><td>0.2728</td><td>0.2457</td></tr><tr><td rowspan="3">NIRSplat-S</td><td> $a t t n ( F _ { r g b } , F _ { t x t } )$   $a t t n ( F _ { r g b } , F _ { n i r } )$ </td><td>0.8053 0.8205</td><td>0.8240</td><td>0.8314</td><td>20.0486</td><td>20.2083</td><td>20.9963</td><td>0.2244</td><td>0.2153</td><td>0.2130</td></tr><tr><td> $a t t n ( F _ { r g b } , F _ { n i r } , F _ { t x t } )$ </td><td>0.8268</td><td>0.8311</td><td>0.8421</td><td>20.7182</td><td>21.0169</td><td>21.0814</td><td>0.2070</td><td>0.2071</td><td>0.2080</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## Benchmarks and Discussions

Previous works often struggle under agricultural conditions due to uncertain camera perspectives, insufficient visual cues, and constraints on computational resources, making robust 3D reconstruction particularly challenging. Specifically, 3DGS suffers from limited 2D information (i.e., due to sparse-view setups), leading to an inevitable performance drop (up to -31.9% SSIM, -5.8 PSNR, and +25.2% LPIPS gaps compared to our NIRSplat), as shown in Tab. 2. In the 3-view configuration, CoR-GS shows substantial structural degradation (lowest SSIM score) among recent sparse-view approaches (Mihajlovic et al. 2024; Fan et al. 2024b). Meanwhile, SplatFields fails to preserve pixel-level fidelity, resulting in a notable drop in PSNR and suboptimal reconstruction quality. Furthermore, these models exhibit poor performance under extremely limited training budgets (1k iterations), suggesting a lack of inherent robustness. Although InstantSplat addresses these drawbacks, this paradigm is still limited in capturing visual details from challenging agricultural samples (i.e., occlusion, uneven reflection), resulting in up to -2.6% SSIM, -2.5 PSNR, and +7.2% LPIPS loss, compared to Ours. To tackle these issues, we leverage a novel multimodal architecture, NIRSplat, which effectively generalizes agricultural environments. Notably, NIRSplat demonstrates its efficiency and validity by surpassing the performance of previous models that use 12 views despite using only 3 views.

## Ablation Studies

Impact of Additional Modalities. This naturally raises a fundamental question: Do additional modalities consistently yield performance improvements? While additional modalities (NIR, Text) provide rich complementary cues, seamlessly integrating them remains a significant challenge. This difficulty largely stems from inherent modality gaps, spectral discrepancies, and disjoint embedding spaces across RGB, NIR, and textual inputs. To better understand this, we explore various fusion strategies in Tab. 3. NaÂ¨Ä±vely adding NIR signals significantly degrades performance, leading to up to a 9% drop in SSIM. We attribute this to the spectral discrepancy and value distribution mismatch between visible and near-infrared modalities. Conventional fusion techniques (element-wise summation, feature concatenation) result in trivial improvements, failing to resolve the semantic and spatial misalignment. Importantly, this limitation potentially becomes more pronounced when incorporating textual metadata as shown in Tab. 3 and Tab. 4: without proper geometric references, semantic and geometric misalignment between visual and textual features causes suboptimal training (i.e., -1.52% SSIM, -2.7 PSNR, +6.74% LPIPS). To address this, we introduce an effective cross-modal 3D reconstruction method, NIRSplat that leverages a geometry-guided 3D point-based positional encoding (PE) scheme anchoring features from all modalities to a shared 2D projection space. Consequently, NIRSplat facilitates robust alignment for uncertain cross-modal knowledge by allowing the model to leverage the most informative signals from each modality, thereby achieving superior 3D consistency and fidelity.

Table 4: Ablation study of PE (Eq. (3)).
<table><tr><td colspan="2">PE</td><td colspan="3">6 views</td></tr><tr><td>w/o</td><td>w/</td><td>SSIM (â)</td><td>PSNR(â)</td><td>LPIPS (â)</td></tr><tr><td>â</td><td></td><td>0.8159</td><td>18.3132</td><td>0.2745</td></tr><tr><td></td><td>â</td><td>0.8311</td><td>21.0169</td><td>0.2071</td></tr></table>

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 5: Comparison under challenging lighting conditions. P.P. and B.P. denote plain prompts and botanic-aware prompts.

<!-- image-->  
Figure 6: Qualitative visualization in a 3-view setup, demonstrating the results under diverse lighting conditions: Lemon (strong light), Kale (artificial light), Art2 (occlusion), and Cabbage (small object). The red box highlights the semantic loss associated with conventional methods.

Effect of Botanic-Aware Knowledge. To further enhance scene understanding in agricultural domains, we incorporate botanic-aware textual prompts that encode physiological and environmental context (i.e., NDVI, NDWI, chlorophyll levels, growth stages, and lighting conditions), as detailed in the supplement. One might reasonably question the effectiveness of language-based guidance in dense 3D reconstruction, as textual descriptions are often semantically abstract and may lack spatial precision. To address this concern, we conduct an ablation study comparing plain prompts with botanic-aware prompts that explicitly embed spectral and biological indices. As shown in Fig. 5, botanic-aware prompts lead to consistent performance gains, with improvements compared to plain prompts under challenging scenarios. These results indicate that botanic-aware prompts, unlike plain prompts, act as highlevel priors that reinforce correlations between NIR responses and botanical states, guiding cross-modal attention toward semantically and structurally relevant regions and improving reconstruction fidelity under ambiguity or occlusion.

## Qualitative Analyses

We qualitatively evaluate our method under four challenging agricultural scenarios: (i) Strong sunlight (Lemon), (ii) artificial lighting (Kale, Art2), (iii) occlusion (Art2), and (iv) small objects (Cabbage), with Blueberry serving as a moderate-complexity reference (see Fig. 6). Conventional methods (e.g., InstantSplat) often fail to preserve semantic and geometric fidelity, showing blurred textures and structural collapse under occlusion or extreme lighting. In contrast, NIRSplat achieves clear improvements by leveraging spectral cues (e.g., NIR) and botanic-aware priors, enabling better detail recovery and structural consistency. Notably, NIRSplat recovers saturated regions under strong illumination, resolves fine details in small-scale objects, and maintains coherence in occluded or low-texture areasâdemonstrating its robustness across diverse agricultural conditions.

## Conclusion

Summary. In this work, we introduced the NIRPlant dataset, which incorporates multimodal data from Near-Infrared (NIR), text, and RGB sensors in both indoor and outdoor agricultural environments. By leveraging the unique advantages of NIR and botanical-aware text, we addressed the challenges of 3D reconstruction in agriculture, including uneven lighting, occlusion, and novel perspectives. We also presented NIRSplat, an effective multimodal Gaussian Splatting framework that bridges these modalities through crossattention and strong geometric priors from 3D point-based positional encoding. Importantly, NIRSplat significantly improves scene understanding, leveraging invisible NIR and contextual text knowledge. Through comprehensive experiments, we demonstrated that NIRSplat outperforms stateof-the-art methods, highlighting the potential of multimodal integration for robust agricultural 3D reconstruction.

Limitations and Future Work. We found that additional inputs lead to significant computational overheads, which limit the efficiency of real-time rendering. While our transformerbased approach effectively bridges the multimodality, it suffers from the cost of increased model complexity and capacity. In future work, we aim to address this issue by seamlessly aligning the three different modalities, ensuring more efficient integration and reducing computational overhead.

## References

Alayrac, J.-B.; Donahue, J.; Luc, P.; Miech, A.; Barr, I.; Hasson, Y.; Lenc, K.; Mensch, A.; Millican, K.; Reynolds, M.; et al. 2022. Flamingo: a visual language model for fewshot learning. Advances in neural information processing systems, 35: 23716â23736.

Barron, J. T.; Mildenhall, B.; Tancik, M.; Hedman, P.; Martin-Brualla, R.; and Srinivasan, P. P. 2021. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 5855â5864.

Barron, J. T.; Mildenhall, B.; Verbin, D.; Srinivasan, P. P.; and Hedman, P. 2022. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 5470â5479.

Bian, W.; Wang, Z.; Li, K.; Bian, J.-W.; and Prisacariu, V. A. 2023. Nope-nerf: Optimising neural radiance field with no pose prior. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 4160â4169.

Chen, S.; Li, Y.; and Zhang, G. 2025. OpticFusion: Multi-Modal Neural Implicit 3D Reconstruction of Microstructures by Fusing White Light Interferometry and Optical Microscopy. In Proceedings of International Conference on 3D Vision (3DV).

Cheng, Y.-C.; Lee, H.-Y.; Tulyakov, S.; Schwing, A. G.; and Gui, L.-Y. 2023. Sdfusion: Multimodal 3d shape completion, reconstruction, and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 4456â4465.

Fan, Z.; Cong, W.; Wen, K.; Wang, K.; Zhang, J.; Ding, X.; Xu, D.; Ivanovic, B.; Pavone, M.; Pavlakos, G.; Wang, Z.; and Wang, Y. 2024a. InstantSplat: Unbounded Sparse-view Posefree Gaussian Splatting in 40 Seconds. arXiv:2403.20309.

Fan, Z.; Wen, K.; Cong, W.; Wang, K.; Zhang, J.; Ding, X.; Xu, D.; Ivanovic, B.; Pavone, M.; Pavlakos, G.; et al. 2024b. InstantSplat: Sparse-view SfM-free Gaussian Splatting in Seconds. arXiv preprint arXiv:2403.20309.

Fu, Y.; Liu, S.; Kulkarni, A.; Kautz, J.; Efros, A. A.; and Wang, X. 2024. Colmap-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 20796â20805.

Garbin, S. J.; Kowalski, M.; Johnson, M.; Shotton, J.; and Valentin, J. 2021. Fastnerf: High-fidelity neural rendering at 200fps. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 14346â14355.

Godard, C.; Mac Aodha, O.; and Brostow, G. J. 2017. Unsupervised monocular depth estimation with left-right consistency. In Proceedings of the IEEE conference on computer vision and pattern recognition, 270â279.

Godard, C.; Mac Aodha, O.; Firman, M.; and Brostow, G. J. 2019. Digging into self-supervised monocular depth estimation. In Proceedings of the IEEE/CVF international conference on computer vision, 3828â3838.

Hao, J.; Liu, J.; Li, J.; Pan, W.; Chen, R.; Xiong, H.; Sun, K.;Lin, H.; Liu, W.; Ding, W.; et al. 2022. AI-enabled Automatic

Multimodal Fusion of Cone-beam CT and Intraoral Scans for Intelligent 3D Tooth-bone Reconstruction and Clinical Applications. arXiv preprint arXiv:2203.05784.

In Lee, D.; Park, H.; Seo, J.; Park, E.; Park, H.; Dam Baek, H.; Sangheon, S.; Kim, S.; et al. 2024. EditSplat: Multi-View Fusion and Attention-Guided Optimization for View-Consistent 3D Scene Editing with 3D Gaussian Splatting. arXiv e-prints, arXivâ2412.

Izadi, S.; Kim, D.; Hilliges, O.; Molyneaux, D.; Newcombe, R.; Kohli, P.; Shotton, J.; Hodges, S.; Freeman, D.; Davison, A.; et al. 2011. Kinectfusion: real-time 3d reconstruction and interaction using a moving depth camera. In Proceedings of the 24th Annual ACM Symposium on User Interface Software and Technology, 559â568.

Kerbl, B.; Kopanas, G.; Leimkuhler, T.; and Drettakis, G. Â¨ 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics, 42(4).

Knapitsch, A.; Park, J.; Zhou, Q.-Y.; and Koltun, V. 2017. Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction. ACM Transactions on Graphics, 36(4).

Lee, S. H.; Li, Y.; Ke, J.; Yoo, I.; Zhang, H.; Yu, J.; Wang, Q.; Deng, F.; Entis, G.; He, J.; et al. 2024. Parrot: Pareto-optimal multi-reward reinforcement learning framework for text-toimage generation. In European Conference on Computer Vision, 462â478. Springer.

Leroy, V.; Cabon, Y.; and Revaud, J. 2024a. Grounding Image Matching in 3D with MASt3R. In European Conference on Computer Vision (ECCV), 71â91.

Leroy, V.; Cabon, Y.; and Revaud, J. 2024b. Grounding Image Matching in 3D with MASt3R. arXiv:2406.09756.

Li, J.; Li, D.; Savarese, S.; and Hoi, S. 2023a. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, 19730â19742. PMLR.

Li, J.; Li, D.; Xiong, C.; and Hoi, S. 2022a. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International conference on machine learning, 12888â12900. PMLR.

Li, K.; Bian, J.-W.; Castle, R.; Torr, P. H.; and Prisacariu, V. A. 2023b. Mobilebrick: Building lego for 3d reconstruction on mobile devices. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 4892â4901.

Li, K.; Pham, T.; Zhan, H.; and Reid, I. 2018. Efficient dense point cloud object reconstruction using deformation vector fields. In Proceedings of the European Conference on Computer Vision (ECCV), 497â513.

Li, L. H.; Zhang, P.; Zhang, H.; Yang, J.; Li, C.; Zhong, Y.; Wang, L.; Yuan, L.; Zhang, L.; Hwang, J.-N.; et al. 2022b. Grounded language-image pre-training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 10965â10975.

Lin, C.-H.; Kong, C.; and Lucey, S. 2018. Learning efficient point cloud generation for dense 3d object reconstruction. In proceedings of the AAAI Conference on Artificial Intelligence, volume 32.

Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023a. Visual instruction tuning. Advances in neural information processing systems, 36: 34892â34916.

Liu, X.; Li, Y.; Teng, Y.; Bao, H.; Zhang, G.; Zhang, Y.; and Cui, Z. 2023b. Multi-modal neural radiance field for monocular dense slam with a light-weight tof sensor. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 1â11.

Liu, Y.; Wang, T.; Zhang, X.; and Sun, J. 2022. PETR: Position embedding transformation for multi-view 3d object detection. In European Conference on Computer Vision (ECCV), 531â548. Springer.

Martin-Brualla, R.; Radwan, N.; Sajjadi, M. S. M.; Barron, J. T.; Dosovitskiy, A.; and Duckworth, D. 2021. NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Mihajlovic, M.; Prokudin, S.; Tang, S.; Maier, R.; Bogo, F.; Tung, T.; and Boyer, E. 2024. Splatfields: Neural gaussian splats for sparse 3d and 4d reconstruction. In European Conference on Computer Vision, 313â332. Springer.

Nguyen, A.-D.; Choi, S.; Kim, W.; and Lee, S. 2019. GraphXconvolution for point cloud deformation in 2D-to-3D conversion. In Proceedings of the IEEE/CVF International conference on computer vision, 8628â8637.

Oh, G.; Jeong, J.; Kim, S.; Byeon, W.; Kim, J.; Kim, S.; and Kim, S. 2024. Mevg: Multi-event video generation with text-to-video models. In European Conference on Computer Vision, 401â418. Springer.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, 8748â8763. PmLR.

Roh, W.; Jung, H.; Kim, J. W.; Lee, S.; Yoo, I.; Lugmayr, A.; Chi, S.; Ramani, K.; and Kim, S. 2024. CATSplat: Context-Aware Transformer with Spatial Guidance for Generalizable 3D Gaussian Splatting from A Single-View Image. arXiv preprint arXiv:2412.12906.

Schonberger, J. L.; and Frahm, J.-M. 2016. Structure-from-Â¨ Motion Revisited. In Conference on Computer Vision and Pattern Recognition (CVPR).

Shu, C.; Deng, J.; Yu, F.; and Liu, Y. 2023. 3DPPE: 3D Point Positional Encoding for Transformer-based Multi-Camera 3D Object Detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 3580â3589.

Sinha, A.; Unmesh, A.; Huang, Q.; and Ramani, K. 2017. Surfnet: Generating 3d shape surfaces using deep residual networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, 6040â6049.

Stotko, P.; Weinmann, M.; and Klein, R. 2019. Albedo estimation for real-time 3D reconstruction using RGB-D and IR data. ISPRS journal of photogrammetry and remote sensing, 150: 213â225.

Sun, L. C.; Bhatt, N. P.; Liu, J. C.; Fan, Z.; Wang, Z.; Humphreys, T. E.; and Topcu, U. 2024. Mm3dgs slam: Multimodal 3d gaussian splatting for slam using vision, depth,

and inertial measurements. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 10159â 10166. IEEE.

Toschi, M.; De Matteo, R.; Spezialetti, R.; De Gregorio, D.; Di Stefano, L.; and Salti, S. 2023. Relight my nerf: A dataset for novel view synthesis and relighting of real world objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 20762â20772.

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Å.; and Polosukhin, I. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Voynov, O.; Bobrovskikh, G.; Karpyshev, P.; Galochkin, S.; Ardelean, A.-T.; Bozhenko, A.; Karmanova, E.; Kopanev, P.; Labutin-Rymsho, Y.; Rakhimov, R.; et al. 2023. Multisensor large-scale dataset for multi-view 3D reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 21392â21403.

Wang, F.; Rakotosaona, M.-J.; Niemeyer, M.; Szeliski, R.; Pollefeys, M.; and Tombari, F. 2024a. UniSDF: Unifying Neural Representations for High-Fidelity 3D Reconstruction of Complex Scenes with Reflections. In Advances in Neural Information Processing Systems, volume 37, 3157â3184. Curran Associates, Inc.

Wang, S.; Leroy, V.; Cabon, Y.; Chidlovskii, B.; and Revaud, J. 2024b. DUSt3R: Geometric 3D Vision Made Easy. In CVPR.

Wang, Z.; Bovik, A.; Sheikh, H.; and Simoncelli, E. 2004. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4): 600â612.

Wu, T.; Zhang, J.; Fu, X.; Wang, Y.; Ren, J.; Pan, L.; Wu, W.; Yang, L.; Wang, J.; Qian, C.; Lin, D.; and Liu, Z. 2023. OmniObject3D: Large-Vocabulary 3D Object Dataset for Realistic Perception, Reconstruction and Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 803â814.

Xiong, B.; Li, Z.; and Li, Z. 2024. GauU-Scene: A Scene Reconstruction Benchmark on Large Scale 3D Reconstruction Dataset Using Gaussian Splatting. arXiv:2401.14032.

Yan, Z.; Mazzacca, G.; Rigon, S.; Farella, E. M.; Trybala, P.; Remondino, F.; et al. 2023. NeRFBK: a holistic dataset for benchmarking NeRF-based 3D reconstruction. International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 48(1): 219â226.

Yang, L.; Kang, B.; Huang, Z.; Xu, X.; Feng, J.; and Zhao, H. 2024. Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data. In CVPR.

Yu, X.; Xu, M.; Zhang, Y.; Liu, H.; Ye, C.; Wu, Y.; Yan, Z.; Zhu, C.; Xiong, Z.; Liang, T.; et al. 2023. Mvimgnet: A large-scale dataset of multi-view images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9150â9161.

Yu, Z.; Chen, A.; Huang, B.; Sattler, T.; and Geiger, A. 2024. Mip-Splatting: Alias-free 3D Gaussian Splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 19447â19456.

Zhang, H.; Li, X.; and Bing, L. 2023. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858.

Zhang, J.; Li, J.; Yu, X.; Huang, L.; Gu, L.; Zheng, J.; and Bai, X. 2024. CoR-GS: sparse-view 3D Gaussian splatting via co-regularization. In European Conference on Computer Vision, 335â352. Springer.

Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang, O. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 586â595.

Zhu, X.; Su, W.; Lu, L.; Li, B.; Wang, X.; and Dai, J. 2020. Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159.

Zollhofer, M.; Stotko, P.; G Â¨ orlitz, A.; Theobalt, C.; NieÃner, Â¨ M.; Klein, R.; and Kolb, A. 2018. State of the art on 3D reconstruction with RGB-D cameras. In Computer graphics forum, volume 37, 625â652. Wiley Online Library.

# Reconstruction Using the Invisible: Intuition from NIR and Metadata for Enhanced 3D Gaussian Splatting

Supplementary Material

## Motivation and Practical Value

Incorporating additional modalities (e.g., NIR, LiDAR, and botanical metadata) poses challenges, as they may not always be feasible in general-purpose applications. However, our focus is not on general-purpose 3D reconstruction but on automated smart farmingâan area where such infrastructure is already being adopted. In commercial precision agriculture, multi-spectral cameras and LiDAR sensors are now standard, with associated metadata (e.g., NDVI, NDWI) often computed automatically by embedded software. In scenarios where high-fidelity reconstruction and crop monitoring are critical $( i . e .$ , phenotyping facilities), we argue that using these modalities is both practical and justified, especially compared to the cost of inaccurate measurements. To be clear, we are not advocating for universal deployment of all modalities, but rather demonstrating that, when such rich data is available, our framework can effectively leverage it to achieve robust and accurate reconstruction under challenging conditions $( e . g .$ , occlusions, uneven lighting). It is noteworthy that as agricultural technology advances, multimodal sensing is becoming increasingly accessible, and methods that can capitalize on it will be essential for future progress.

## Additional Related Works

## Multimodal 3D Reconstruction

Multimodal 3D Reconstruction methods leverage the complementary strengths of multiple data types to produce more accurate reconstructions. Traditional multimodal models that combine RGB image data and depth data have commonly been used to produce dense, high-quality reconstructions of 3D surfaces (Izadi et al. 2011; Zollhofer et al. 2018). The use Â¨ of other data types, including infrared, time-of-flight, white light interferometry, and cone beam computed tomography data, for multimodal 3D Reconstruction has also been explored (Liu et al. 2023b; Stotko, Weinmann, and Klein 2019; Chen, Li, and Zhang 2025; Hao et al. 2022).

Recently, neural network-based multimodal 3D reconstruction methods have been developed. Differentiable volume rendering techniques from single-modal Reconstruction have been applied to multimodal 3D reconstruction (Liu et al. 2023b; Sun et al. 2024). Modern data encoders such as BERT and CLIP allow 3D scene completion and generation guided by a combination of text, image, and incomplete scene data (Cheng et al. 2023).

## Preliminary

3D Gaussian Splatting. We build upon the foundational 3D Gaussian Splatting (3DGS) method (Kerbl et al. 2023), which represents a 3D scene using a set of anisotropic Gaussian primitives. As shown in Eq. (7), each primitive is parameterized by a 3D position $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ , an opacity value Î±, a full covariance matrix $\pmb { \Sigma } \in \dot { \mathbb { R } } ^ { 3 \times 3 }$ encoding spatial uncertainty, and view-dependent color modeled via spherical harmonics (SH) as $\begin{array} { r } { \dot { c } ( \mathbf { d } ) = \sum _ { i } { c _ { i } \mathcal { B } _ { i } ( \mathbf { d } ) } } \end{array}$ , where $B _ { i }$ are the SH basis functions.

$$
{ \bf G } ( { \bf p } , \alpha , { \pmb \Sigma } ) = \alpha \exp ( - \frac { 1 } { 2 } ( { \bf p } - { \bf x } ) ^ { T } { \Sigma } ^ { - 1 } ( { \bf p } - \mu ) ) .\tag{7}
$$

Unlike volumetric radiance fields $( e . g .$ ., NeRF) or voxel-based approaches, 3DGS eliminates the need for expensive ray marching by directly projecting the Gaussians into screen space, enabling efficient rasterization and differentiable rendering. This results in significantly improved rendering speed, compact representation, and high fidelity for real-time novel view synthesis, making it highly suitable for dynamic or resource-constrained applications.

Multi-View Stereo. MASt3R incorporates a dense local feature head and a fast reciprocal nearest-neighbor matching scheme and optimizes the pixel-aligned point maps P directly from raw images. Unlike traditional 2D matching approaches, MASt3R treats matching as an intrinsically 3D task by regressing point maps in a shared coordinate frame, thus enabling robust correspondence even under extreme viewpoint variations. In this paper, building on the feed-forward Multi-View Stereo framework MASt3R (Leroy, Cabon, and Revaud 2024a), we introduce a tri-modal architecture NIRSplat that takes as input aligned RGB frames $\mathcal { T } _ { r q b } ^ { N } \in \mathbb { R } ^ { N \times \dot { H } \times W \times 3 }$ single-channel NIR frames $\mathcal { T } _ { n i r } ^ { N } \in \mathbb { R } ^ { N \times H \times W \times 1 }$ , and contextual text prompts $\mathcal { T } \in \mathbb { R } ^ { L \times \ddot { C } }$ . Dedicated encoders transform each modality into latent sequences; $\mathbf { F } _ { r g b } = \{ f _ { r g b } ^ { i } \} _ { i = 1 } ^ { N } ,$ $\mathbf { F } _ { n i r } = \{ f _ { n i r } ^ { i } \} _ { i = 1 } ^ { N }$ , and $\mathbf { F } _ { t x t } = \{ f _ { t x t } ^ { i } \} _ { i = 1 } ^ { L }$ are subsequently fused by a deformable-cross-attention Transformer (Vaswani et al. 2017; Zhu et al. 2020). This fusion module aligns spectral $( \mathrm { R G B }  \mathrm { N I R } )$ and semantic (vision â language) cues in a shared token space, letting attention weights act as adaptive modality gates: noisy RGB pixels under adverse lighting conditions are down-weighted, while light-robust NIR edges and text-driven botanical priors are amplified.

Loss function. Motivated by recent studies (Fan et al. 2024b; Leroy, Cabon, and Revaud 2024b; Wang et al. 2024b), we directly penalize between the predicted point maps P and the pseudo ground truths $\hat { \mathbf { P } }$ from MASt3R using the following regression loss term:

$$
\mathcal { L } _ { r e g } = \Big \| \frac { 1 } { z _ { i } } \cdot \mathbf { P } _ { v , 1 } - \frac { 1 } { z _ { i } } \cdot \hat { \mathbf { P } } _ { v , 1 } \Big \| ,\tag{8}
$$

where $v \in \{ 1 , 2 \} , z$ denote corresponding views and normalization factors. The predicted Gaussian means $\mathbf { P } _ { v , 1 }$ are obtained via our NIRSplat framework, which enables efficient semantic grounding in 3D space. Plus, we optimize the pixel-aligned confidence score $O _ { v , 1 } ^ { i }$ to improve scene understanding through the following objective:

$$
\mathcal { L } _ { c o n f } = \sum _ { v \in 1 , 2 } \sum _ { i \in D ^ { v } } O _ { v , 1 } ^ { i } \cdot \mathcal { L } ( v , i ) - \alpha \cdot \log O _ { v , 1 } ^ { i } .\tag{9}
$$

Table 5: Botanic-aware Prompt Engineering.
<table><tr><td>Botanic-aware Prompt Engineering</td></tr><tr><td>Y </td></tr><tr><td>The provided data are as follows: Image Description:{img_description}</td></tr><tr><td>NIR Information:{NDVI, NDWI, chlorophyll index} Weather Information: {temp, dew, humidity, precip, precipprob, cloudcover, solarradiation, uvindex,</td></tr><tr><td>windgust, windspeed, visibility}</td></tr><tr><td>Instructions: Do not simply concatenate the inputs. Instead, synthesize them into a natural and cohesive narrative.</td></tr><tr><td>relevant components.</td></tr><tr><td>&quot;81%&quot;).</td></tr><tr><td>structures.</td></tr><tr><td>â¢Limit your output to a paragraph (max 200 words) in fluent and formal English.</td></tr><tr><td>â¢Do not include explanations, metadata, or any extra text. Output only the final paragraph.</td></tr></table>

The final rewritten description is:

where L(v, i) denotes Eq. (8), and Î± is a balancing weight. The first term encourages accurate predictions in confident regions, while the second regularizes overconfidence via a log penalty. This formulation enables the model to selectively focus on spatially reliable regions and suppress uncertain noise, resulting in substantial improvements in both semantic fidelity and geometric precision.

## Botanic-aware Prompt Engineering

To facilitate robust scene understanding under challenging agricultural conditions, we design a botanic-aware prompt engineering strategy (Tab. 5) that synthesizes complementary visual, physiological, and environmental modalities into a unified textual representation. Our prompt integrates (i) image descriptions highlighting structural features such as occlusion, overlapping leaves, or lighting variation, (ii) NIRderived physiological cues including NDVI, NDWI, and chlorophyll index, and (iii) weather conditions such as humidity, solar radiation, or wind gusts that may influence plant appearance and geometry. Unlike naive concatenation of metadata, the prompt generation process encourages the creation of a fluent paragraph that semantically aligns these modalities, enabling the model to infer cross-modal relationships and guide attention toward informative regions. In particular, collaborating with NIR signals (i.e., highly sensitive to internal plant states) offers deep contextual cues that enrich understanding of vegetation health, moisture levels, and pigment concentration. Importantly, this multimodal prompt not only anchors spatial features but also enhances generalizability across diverse environments, contributing to downstream 3D reconstruction tasks.

## Experimental Configurations.

Implementation Detail. To ensure a fair comparison with our baseline, we adopt a sparse-view 3D reconstruction paradigm using 3, 6, and 12 input views. For each setting, we perform 200 and 1,000 iterations for training and 500 iterations for test-view optimization, respectively, maintaining consistency across all experiments. Furthermore, to account for the relatively slower convergence of certain baselines, we extend the training schedules of 3DGS (Kerbl et al. 2023), CoR-GS (Zhang et al. 2024) and SplatFields (Mihajlovic et al. 2024) to 10k and 30k iterations, respectively, thereby providing sufficient optimization steps and mitigating potential underfitting in the sparse-view scenario. Unless otherwise stated, all methods are trained and evaluated using two NVIDIA RTX 6000 ADA GPUs under identical computational conditions.

Evaluation Metrics. For quantitative evaluation, we employ three widely adopted metrics for novel view synthesis, following prior work (Kerbl et al. 2023; Fan et al. 2024a): Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM) (Wang et al. 2004), and Learned Perceptual Image Patch Similarity (LPIPS) (Zhang et al. 2018). Among these, PSNR measures the pixel-wise reconstruction fidelity, SSIM evaluates the perceived structural similarity between the reconstructed and reference images, and LPIPS assesses perceptual quality using deep feature embeddings. To ensure a consistent and fair comparison under sparse-view conditions, we strictly follow the evaluation protocol of the state-of-the-art InstantSplat (Fan et al. 2024a), including its sparse-view reconstruction settings and test-view sampling strategy.

Table 6: Ablation study on NIRPlant. Text, NIR, and Temp represent RGB-Text Interaction, NIR-RGB Interaction, and Temporal Interaction, respectively. We conduct an ablation study with 3, 6, and 12 view setups and 1000 iterations. Note that bold values indicate the best performance.
<table><tr><td colspan="2">Method</td><td colspan="3">SSIM (â)</td><td colspan="3">PSNR (â)</td><td colspan="3">LPIPS (â)</td></tr><tr><td>Text</td><td>NIR</td><td>3-view</td><td>6-view</td><td>12-view</td><td>3-view</td><td>6-view</td><td>12-view</td><td>3-view</td><td>6-view</td><td>12-view</td></tr><tr><td rowspan="2"></td><td></td><td>0.7984</td><td>0.8126</td><td>0.8134</td><td>18.3849</td><td>18.9233</td><td>19.0333</td><td>0.2797</td><td>0.2689</td><td>0.2438</td></tr><tr><td></td><td>0.8053</td><td>0.8139</td><td>0.8160</td><td>18.8696</td><td>18.7132</td><td>19.1866</td><td>0.2765</td><td>0.2728</td><td>0.2457</td></tr><tr><td rowspan="2">â</td><td>â</td><td>0.8205</td><td>0.8240</td><td>0.8314</td><td>20.0486</td><td>20.2083</td><td>20.9963</td><td>0.2244</td><td>0.2153</td><td>0.2130</td></tr><tr><td>â</td><td>0.8268</td><td>0.8311</td><td>0.8421</td><td>20.7182</td><td>21.0169</td><td>21.0814</td><td>0.2070</td><td>0.2071</td><td>0.2080</td></tr></table>

Table 7: Ablation study of metadata. We train our model with six view setups and 1000 iterations. Note that bold values indicate the best performance.
<table><tr><td colspan="2">Metadata</td><td colspan="3">6 views</td></tr><tr><td>w/o</td><td>w/</td><td>SSIM (â)</td><td>PSNR (â)</td><td>LPIPS (â)</td></tr><tr><td>â</td><td></td><td>0.8299</td><td>20.8694</td><td>0.2172</td></tr><tr><td></td><td>â</td><td>0.8311</td><td>21.0169</td><td>0.2071</td></tr></table>

## Additional Experiments.

Ablation of modalities. We ablate our proposed method, NIRSplat, to validate the contribution of each knowledge source (i.e., NIR and Text) under different view configurations (i.e., 3, 6, and 12 views) with 1000 iterations, as shown in Tab. 6. First, incorporating only the text modality already leads to notable improvements over the baseline InstantSplat across all metrics. This gain is attributed to our botanic-aware textual guidance, which enhances scene understanding through weighted attention and semantically rich alignment. Second, the inclusion of the NIR modality also contributes positively, particularly in low-visibility and occlusion-heavy conditions commonly found in agricultural environments. It yields improved perceptual quality and spatial consistency across all views. Most importantly, when both modalities (i.e., text and NIR) are combined, the model achieves the best overall performance: +2.8% SSIM, +1.7% PSNR, and â5.5% LPIPS improvement over the baseline at 3-view, with consistent gains across all view settings. This highlights the synergistic effect of multimodal integration in our framework. Overall, NIRSplat effectively leverages complementary multimodal cues to overcome the limitations of conventional methods, demonstrating its superiority in robustness, accuracy, and perceptual fidelity.

Delving into botanic-aware prompts. We observed that in Tab. 6 RGB-only methods struggle to reconstruct finegrained agricultural structures under challenging lighting conditions, mainly due to overexposure and low illumination. Here, we present an in-depth analysis of how botanicaware knowledge significantly enhances scene understanding in novel agricultural environments. First, the NIR modality contributes significantly by operating in a distinct spectral range, enabling consistent structural recovery even under harsh lighting. This yields notable gains across all metrics (see +NIR in Tab. 6). Crucially, we further boost the effectiveness of text guidance through botanic-aware prompt engineering, where prompts explicitly encode spectral and environmental metadata (e.g., NDVI, lighting conditions). As validated in Tab. 7, incorporating metadata yields measurable improvements: +0.12 SSIM, +0.15 PSNR, and â0.0051 LPIPS, demonstrating that environmental cues are not only semantically meaningful but also practically beneficial. Furthermore, our method outperforms landmark vision-language models such as CLIP and BLIP in multi-view reconstruction tasks (see Tab. 8). Notably, BLIP-2 + Metadata achieves the best overall performance across all metrics, indicating the synergy between domain-specific prompting and adaptive attention. During weighted cross-attention, NIRSplat dynamically assigns higher attention to prompt tokens enriched by these metadata-driven priors, thereby enhancing modality alignment and scene-level reasoning. This selective focus enables the model to effectively decode and integrate FNRâthe fused feature from NIR and RGBâresulting in perceptually faithful and structurally accurate reconstructions.

Table 8: Comparison with landmark VLMs. We train our model with six view setups and 1000 iterations. Note that bold values indicate the best performance.
<table><tr><td rowspan="2">Method</td><td colspan="3">6 views</td></tr><tr><td>SSIM (â)</td><td>PSNR (â)</td><td>LPIPS (â)</td></tr><tr><td>CLIP (Radford et al. 2021)</td><td>0.8281</td><td>20.2083</td><td>0.2080</td></tr><tr><td>BLIP (Li et al. 2022a)</td><td>0.8208</td><td>21.0330</td><td>0.2090</td></tr><tr><td>BLIP-2 (Li et al. 2023a)</td><td>0.8311</td><td>21.0169</td><td>0.2071</td></tr></table>

## Additional Qualitative Analyses

In this section, we qualitatively demonstrate that our proposed approach effectively addresses the core challenges of agricultural scene reconstruction by leveraging invisible yet informative multimodal cues, as shown in Fig. 7. Specifically, we introduce four representative and challenging scenarios in which existing methods consistently fail to preserve key visual properties such as texture, shape, and volume, primarily due to ambiguous or missing semantic clues (highlighted in red dotted boxes). Agricultural environments inherently present unique obstacles, including incomplete or skewed viewpoints, extreme lighting conditions (e.g., overexposure or underexposure), and heavy occlusion from overlapping foliage or dense vegetation. These factors often result in non-trivial failures during 3D reconstruction, as illustrated in Fig. 7. First, we observe that existing methods are highly sensitive to uncontrolled lighting (particularly in cases of whiteout caused by strong sunlight), where they tend to oversmooth or completely wash out structural content. Second, excessive occlusion often introduces redundancy and ambiguity in depth reasoning, leading to artifacts or duplicated geometries (see the red box in Art2). Third, reconstructing fine-scale texture and geometryâsuch as small and dense crops like cabbagesâremains a significant challenge for prior approaches due to their limited spatial resolution and inability to preserve fine-grained surface detail, often resulting in blurry or semantically inconsistent outputs. To overcome these limitations, we propose NIRPlant and our full model NIRSplat, which jointly incorporate NIR and text-guided prompts to adaptively handle challenging conditions. Our method robustly compensates for missing semantic cues through modality-aware attention and is able to synthesize coherent, high-fidelity 3D structures across diverse lighting, occlusion, and viewpoint variations. These results highlight the practical benefits of multimodal fusion in advancing scene understanding in real-world agricultural scenarios.

<!-- image-->  
Figure 7: Qualitative visualization in a 3-view setup, demonstrating the results under diverse lighting conditions: Lemon (strong light), Kale (artificial light), Art2 (occlusion), and Cabbage (small object). The comparison highlights the semantic loss when using conventional methods, with a marked improvement achieved by our novel approach, NIRSplat. Zoom in for better visualization.

## Licenses

Our collected data is under the CC-BY-4.0 (https:// creativecommons.org/licenses/by/4.0/legalcode.en) license. In addition, the dataset shall be used only for non-commercial research and educational purposes.

## Acknowledgements

This research was supported by Culture, Sports and Tourism R&D Program through the Korea Creative Content Agency grant funded by the Ministry of Culture, Sports and Tourism (Project Name: International Collaborative Research and Global Talent Development for the Development of Copyright Management and Protection Technologies for Generative AI, Project Number: RS-2024-00345025) and the National Institute of Food and Agriculture of the US Department of Agriculture (grant number 2024-67021-42528).