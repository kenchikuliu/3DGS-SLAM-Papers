# Seeing Through Reflections: Advancing 3D Scene Reconstruction in Mirror-Containing Environments with Gaussian Splatting

Zijing Guo1, Yunyang Zhao1, and Lin Wang2

1School of Automation and Intelligent Sening, Shanghai Jiao Tong University   
2Ningbo Artificial Intelligence Institute, Shanghai Jiao Tong University

## Abstract

Mirror-containing environments pose unique challenges for 3D reconstruction and novel view synthesis (NVS), as reflective surfaces introduce view-dependent distortions and inconsistencies. While cutting-edge methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) excel in typical scenes, their performance deteriorates in the presence of mirrors. Existing solutions mainly focus on handling mirror surfaces through symmetry mapping but often overlook the rich information carried by mirror reflections. These reflections offer complementary perspectives that can fill in absent details and significantly enhance reconstruction quality. To advance 3D reconstruction in mirror-rich environments, we present MirrorScene3D, a comprehensive dataset featuring diverse indoor scenes, 1256 high-quality images, and annotated mirror masks, providing a benchmark for evaluating reconstruction methods in reflective settings. Building on this, we propose ReflectiveGS, an extension of 3D Gaussian Splatting that utilizes mirror reflections as complementary viewpoints rather than simple symmetry artifacts, enhancing scene geometry and recovering absent details. Experiments on MirrorScene3D show that ReflectiveGaussian outperforms existing methods in SSIM, PSNR, LPIPS, and training speed, setting a new benchmark for 3D reconstruction in mirror-rich environments.

## 1 Introduction

Mirror-containing scenes are ubiquitous in modern environments, from homes with reflective furniture to buildings with glass facades. These reflective surfaces, while visually striking, pose significant challenges for 3D reconstruction and novel view synthesis (NVS), both of which are critical for applications in virtual reality, robotics, and computer graphics. Unlike diffuse surfaces, mirrors introduce view-dependent distortions and occlusions, making it difficult for traditional reconstruction pipelines to accurately capture and render such scenes [12, 19, 38, 41].

In recent years, Neural Radiance Fields (NeRF) [24] and 3D Gaussian Splatting (3DGS) [14] have revolutionized 3D reconstruction. NeRF models density and view-dependent colors of scene points using neural implicit fields, while 3DGS represents scenes explicitly using Gaussian ellipsoids with anisotropic covariance matrices. These methods achieve remarkable performance in typical scenes [47, 31, 7], with 3DGS further excelling in real-time rendering and visual fidelity [49, 36, 16]. However, their effectiveness diminishes in mirror-containing environments, where view-inconsistent reflections disrupt reconstruction pipelines. Existing methods, such as Mirror-NeRF [46] and TraM-NeRF [8], address this issue by focusing on rendering the mirror surface itself, using techniques like ray tracing and symmetry mapping to generate reflections. Similarly, 3DGS-based approaches, including MirrorGaussian [18], have improved visual quality by blending mirror and non-mirror regions. While effective for mirror rendering, these methods neglect a critical aspect: the valuable information embedded in mirror reflections. Reflected content offers complementary viewpoints that can fill in absent details of the real scene, a potential that has been rarely explored in previous work.

<!-- image-->  
Figure 1: Comparison of Methods and Rendering Performance of Vanilla3DGS, Mirror3DGS, and ReflectiveGS (Ours) on the âCupâ Scene. Vanilla 3DGS is unable to capture viewpoints on the mirror side, leading to incomplete reconstructions. Mirror-3DGS focuses on refining mirror surface details by distinguishing mirror and non-mirror regions using a blue mask. While this prevents interference from virtual 3D Gaussians, it limits the methodâs ability to utilize reflection-based information for reconstruction. ReflectiveGS (Ours) incorporates supplementary mirror data to recover occluded details, enhancing reconstruction quality by integrating mirror reflections into joint training. Compared to other methods, our approach achieves more realistic and detailed renderings while maintaining competitive training times.

To address this gap, we introduce MirrorScene3D, a new dataset specifically designed for mirroraugmented reconstruction (Using information contained in mirror to accomplish object-cnetric 3D Reconstruction, One example in Figure 1). It contains five diverse scenes with approximately 1,000 images and corresponding mirror masks (see Table 1), providing a benchmark for evaluating reconstruction methods in reflective environments.

Building on this, we propose ReflectiveGS, a novel extension of 3D Gaussian Splatting that explicitly integrates mirror reflection data into the reconstruction process. Rather than discarding reflections or treating mirrors as simple symmetry artifacts, ReflectiveGS leverages reflected content as additional viewpoints, enriching scene geometry and filling in absent details. Our pipeline incorporates multi-view object-centric images with mirror masks and introduces a symmetry consistency loss to jointly optimize real and reflected objects, ensuring structural coherence. Our method achieves improvements across all metrics compared to the baseline methods, with an average 30% SSIM increase, 2.0 dB PSNR boost, and 20% LPIPS reduction, while maintaining competitive training time, balancing visual quality and computational efficiency.

In summary, our contributions are as follows:

â¢ We introduce MirrorScene3D, a novel dataset designed for mirror-augmented 3D reconstruction, offering high-quality multi-view images and mirror masks to benchmark reconstruction

<table><tr><td>Dataset</td><td>Task Focus</td><td>Scene Type</td><td>Scene Numbers Avg. Images Mirror Info</td><td></td><td></td></tr><tr><td>MSD [37]</td><td>Mirror Segmentation</td><td>Independent Images</td><td>-</td><td></td><td>X</td></tr><tr><td>PMD [17]</td><td>Mirror Segmentation</td><td>Independent Images</td><td>-</td><td>-</td><td>X</td></tr><tr><td>Mirror-NeRF[46]</td><td>Mirror Rendering</td><td>Complete Scenes</td><td>5 Synthetic, 4 Real</td><td>200~300</td><td>X</td></tr><tr><td>MirrorScene3D(Ours)</td><td>Mirror Supplementation</td><td>Complete Scenes</td><td>5 Real</td><td>200~300</td><td>â</td></tr></table>

Table 1: Comparison of Our Benchmark with Existing Datasets. Our dataset (MirrorScene3D) is specifically designed to validate algorithms for 3D scene reconstruction that utilize mirror information and address the supplementation of absent details. It includes five comprehensive real-world scenes, with an average scale comparable to previous datasets, while also integrating exploitable mirror data for augmented reconstruction accuracy.

methods in reflective environments.

â¢ We propose ReflectiveGS, an approach that fully integrates mirror reflection data into 3D reconstruction, leveraging reflections as complementary viewpoints to enhance reconstruction quality in mirror-rich scenes.

â¢ We develop a novel optimization strategy based on symmetry consistency loss, enabling the joint training of real and reflected objects to ensure structural coherence and detail preservation.

## 2 Related Works

This section reviews two categories of work closely related to our research: novel view synthesis and mirror-aware reconstruction.

## 2.1 Novel View Synthesis

NVS (Novel View Synthesis) generates new images from novel viewpoints using a limited set of known perspectives [4, 32, 23]. By learning the structure of objects or scenes from multiple angles, it synthesizes images from new viewpoints. The introduction of NeRF (Neural Radiance Fields) in 2020 [24] marked a significant breakthrough. NeRF uses MLPs to estimate density and viewdependent colors for points via ray marching and volume rendering, producing impressive visuals. Research on NeRF has advanced rapidly, focusing on speed, quality, and handling challenging scenes [25, 42, 5, 2, 1, 43, 20, 26]. However, its reliance on dense sampling and computationally intensive ray marching remains a challenge, particularly in real-time applications or incomplete viewpoints.

The emergence of 3DGS (3D Gaussian Splatting) [14] in 2023 has further advanced the development of this field [33, 44, 39, 15, 29, 10]. In contrast to NeRF, 3DGS explicitly represents 3D scenes using anisotropic Gaussians and leverages a tile-based rasterizer that allows Î±-blending. This enables substantial improvements in training efficiency, visual quality, and real-time rendering, making it well-suited for interactive applications. Furthermore, 3DGS is applied in various domains [34], such as scene understanding and segmentation [28, 40, 11, 3, 48], SLAM (Simultaneous Localization and Mapping) [35, 21, 13, 45, 9], and more.

Despite its advantages, 3DGS struggles with novel view synthesis under incomplete viewpoints. To overcome this, we incorporate mirror reflection as supplementary information and apply mirror symmetry to reconstruct absent viewpoints within the 3DGS framework.

<!-- image-->  
Figure 2: Scenes from Our Dataset. Our dataset captures $0 ^ { \circ } - 1 8 0 ^ { \circ }$ multi-view images from the opposite side of the mirror, with the central object as the focus. Occlusion regions are set, which are only visible through the mirror reflection and not directly captured.

## 2.2 Mirror-Aware Reconstruction

Mirrors are a crucial source of supplementary information in our research. Existing studies on mirror scenes primarily focus on rendering mirror regions, where specular reflections cause multiview inconsistencies, leading to blurriness or aliasing. NeRF-based methods have attempted to address these issues: Ref-NeRF [30] reparameterizes directional MLPs for improved view-dependent results, while Mirror-NeRF [46] models reflection probability to blend camera and reflected rays. However, these methods suffer from long runtimes, limiting their efficiency and quality.

Recent 3DGS-based methods offer new perspectives. Mirror-3DGS [22] introduces mirror attributes into 3DGS to derive mirror transformation, enhancing the realism of mirror rendering through two-stage training process. MirrorGaussian [18] proposes a dual-rendering approach that enables differentiable rasterization of both the real-world 3D Gaussians and their mirrored counterparts.

However, these methods discard mirror content and rely solely on physical symmetry to generate mirror rendering. In contrast, our approach captures and leverages mirror reflections to supplement object reconstruction, achieving more detailed results and better visual quality.

## 3 MirrorScene3D Construction

## 3.1 Data Construction

We conduct MirrorScene3D, a dataset designed to leverage mirror reflections for supplementing absent details in scene reconstruction. Existing mirror-related datasets mainly focus on mirror segmentation (e.g., MSD [37], PMD [17]) or rendering effects of mirror regions (e.g., Mirror-NeRF [46]), but none address the challenge of utilizing mirror reflections to recover occluded viewpoints.

Our dataset consists of diverse indoor scenes for validating our method. It targets static, objectcentric tasks, where a mirror is placed on one side of the object, and the camera captures 180â¦ multi-view images from the opposite side. This setup ensures that occluded object regions are only visible through mirror reflections. To enable quantitative evaluation, we also capture complete multi-view images after removing the mirrors, providing ground truth for previously occluded areas. Table 1 presents a comparison with related datasets.

<!-- image-->  
Figure 3: Overview of the Method. Our method takes images with viewpoint occlusions as input and utilizes three key steps to leverage mirror reflection information for augmented 3D scene reconstruction with more detail. First, we introduce a learnable mirror factor to represent the probability of mirror Gaussians, updating it via a loss function that compares the predicted mask with the ground truth. This allows us to fit the mirror equation and compute the corresponding symmetry metric. Next, based on conventional rendering results, we distinguish the Gaussians in front and behind the mirror by evaluating their distances to the mirror plane. The computed symmetry metric is then used to perform the symmetry mapping operation. To constrain training, we introduce a symmetry consistency loss, which minimizes the difference between the center coordinates of original and flipped Gaussians, ensuring collaborative optimization of Gaussians on both sides of the mirror.

## 3.2 Scene Arrangement and Data Collection

Our dataset includes diverse indoor scenes: residential settings (e.g., a cup, a sofa), toy scenes (e.g., a Psyduck figure, small ducks), office-like environments (e.g., a stack of books), and kitchen scenes (e.g., apples, bananas, leafy greens). The objects span various categories, with geometric diversity ranging from regular cuboidal shapes (e.g., sofas, books) to irregular surfaces (e.g., toys, fruits), providing a robust benchmark for reconstruction evaluation.

Viewpoint Occlusion. To define occluded regions, we designate specific areas in each scene: the front pattern of the cup (âCupâ scene), the bottle between bag and sofa (âSofaâ scene), musical notes under Psyduck (âDuckâ scene), text and patterns on book spines (âBookâ scene), and the hawthorn (âFruitâ scene). These areas remain unobserved in the 180â¦ multi-view capture but appear in mirror reflections. Figure 2 illustrates the capture viewpoints. The reconstruction of these occluded regions serves as a key measure of algorithmâs ability to supplement absent details.

Mirror Placement. Mirror placement was standardized across all scenes to ensure consistency and comparability. Mirrors were positioned at a fixed height and distance from objects, maintaining uniform alignment and optimized angles to capture both objects and their surroundings without overlap or distortion. Calibration markers ensured precise positioning, and adjustments were made to ensure reflective regions covered key object features. This systematic setup provides reliable reflection data for evaluating reconstruction algorithms.

Lighting Conditions. Lighting conditions were carefully controlled to maintain consistency while accounting for environmental factors. A combination of natural and artificial light sources was used to replicate indoor settings, with optimized intensity and direction to minimize shadows, glare, and unwanted reflections. Considerations such as room layout, surface reflectivity, and light diffusion ensured uniform illumination across all scenes, enhancing the datasetâs reliability for reconstruction evaluation.

## 3.3 Data Annotation and Preprocessing

Mirror regions were manually annotated using LabelMe, saved as .json files, and converted to .jpg format for binary mask generation. All images were resized to 1280 Ã 720, resulting in a dataset of 1256 images across five diverse indoor scenes, serving as a benchmark for evaluating mirroraugmented reconstruction methods.

A standardized preprocessing pipeline was applied to ensure consistency. Gaussian filtering reduced image noise, while color correction addressed lighting variations. Images were resized for uniformity, and contrast adjustments enhanced visibility, especially in reflective regions, ensuring a well-structured dataset for reconstruction tasks.

## 3.4 Summary

MirrorScene3D is a new dataset designed for mirror-augmented 3D reconstruction, focusing on supplementing absent viewpoints through mirror reflections. It includes 1256 images spanning diverse indoor scenes with standardized mirror placement and lighting. The dataset provides a high-quality benchmark for assessing the integration of mirror reflections in reconstruction tasks.

## 4 Method

Overview. The method overview is illustrated in Figure 3. ReflectiveGS leverages mirror information as supplementary data to improve the reconstruction of objects with absent views caused by occlusions in mirror-containing 3D scenes. The inputs to the method include multi-view images of the scene, corresponding mirror masks, and a sparse point cloud generated through structurefrom-motion (SfM) [27]. While some object features are absent in direct views, they can often be observed in mirror reflections, providing valuable complementary information. Our method is structured around three key components:

â¢ Mirror Plane Estimation: A parameterized mirror plane equation is used to establish the geometric relationship between real objects and their virtual reflections, enabling accurate feature mapping.

â¢ Mirror Symmetry Mapping: Leveraging the mirror plane equation, reflected objects are rendered through symmetry mapping. Depth selection ensures only valid reflected features are incorporated to complement real objects.

â¢ Symmetry-Aware Joint Optimization: A spatial symmetry consistency loss function is introduced to collaboratively optimize the real and reflected components of the point cloud, ensuring alignment and improving detail preservation.

Further details on these components are provided in Section 4.2 (Mirror Plane Estimation), Section 4.3 (Mirror Symmetry Mapping), and Section 4.4 (Symmetry-Aware Joint Optimization). Together, these steps enable ReflectiveGS to effectively utilize mirror reflections, enriching 3D scene reconstruction and addressing the limitations of traditional methods.

## 4.1 Preliminaries

3DGS is a method that explicitly represents 3D scenes with numerous anisotropic 3D Gaussian primitives. Each of them can be described as:

$$
\mathcal { G } _ { i } ( x ) = e x p \{ - \frac { 1 } { 2 } ( x - \mu _ { i } ) ^ { T } \Sigma _ { i } ^ { - 1 } ( x - \mu _ { i } ) \} ,\tag{1}
$$

where $\mu _ { i } \in \mathbb { R } ^ { 3 }$ represents the center position of $\mathcal { G } _ { i } ,$ , and $\Sigma _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ represents the 3D covariance matrix. $\Sigma _ { i }$ also can be decomposed into a rotation matrix $R _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ and a scaling matrix $S _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ as shown in Equation 2.

$$
\Sigma _ { i } = R _ { i } S _ { i } S _ { i } ^ { T } R _ { i } ^ { T } .\tag{2}
$$

The subsequent rasterization process projects the 3D Gaussian primitives onto the 2D screen space, with their shapes and extents determined by the covariance matrices. The final image is then generated by leveraging Î±-Blending to integrate the colors and opacities of multiple Gaussians. Based on the differentiable rasterization principles of 3DGS, the rendering process is expressed as:

$$
\begin{array} { l } { { \displaystyle { \bf C } ( { \bf p } ) = \sum _ { i \in N } c _ { i } ( d ) \alpha _ { i } \mathcal { G } _ { i } ^ { \prime } ( p ) T _ { i } } , } \\ { { \displaystyle \qquad } } \\ { { T _ { i } = \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } \mathcal { G } _ { i } ^ { \prime } ( p ) , } } \\ { { \displaystyle \qquad } } \\ { { \mathcal { G } _ { i } ^ { \prime } ( p ) = e ^ { - \frac 1 2 ( p - \mu _ { i } ^ { \prime } ) ^ { T } \Sigma _ { i } ^ { \prime - 1 } ( p - \mu _ { i } ^ { \prime } ) } , } } \end{array}\tag{3}
$$

where $\mathbf { C } ( \mathbf { p } )$ is the color of a pixel p, $\mathcal { G } _ { i } ^ { \prime } ( \boldsymbol { p } )$ represents the projected 2D Gaussians, N represents the number of sample Gaussians, $c _ { i }$ and $\alpha _ { i }$ represent the color and opacity of the i-th Gaussian respectively.

## 4.2 Mirror Plane Estimation

The foundation of 3D reconstruction in mirror-containing scenes lies in the principle of mirror symmetry. Accurately parameterizing the mirror plane is critical for linking real objects and their virtual counterparts. To achieve this, we introduce a learnable mirror factor $m \in ( 0 , 1 )$ , an attribute of 3D Gaussians that represents the probability of a point belonging to the mirror. The mirror factor is combined with opacity Î±, yielding a modified opacity $\alpha _ { i } ^ { \prime } = m _ { i } \alpha _ { i }$ . The updated rendering equation is expressed as:

$$
\mathbf { C ( p ) } = \sum _ { i \in N } c _ { i } ( d ) \alpha _ { i } ^ { \prime } \mathcal { G } _ { i } ^ { \prime } ( p ) T _ { i } ,\tag{4}
$$

where $\alpha _ { i }$ denotes the probability of a ray intersecting the i-th Gaussian, and $m _ { i }$ adjusts visibility for mirror regions. High mirror factors enhance visibility for mirror regions, while non-mirror regions are rendered nearly transparent. The resulting rendered mirror mask is compared to the ground truth to refine the learning of the mirror factor.

Using the learned mirror factor, regions with high mirror probabilities are identified and used to fit a parameterized mirror plane equation via the RANSAC algorithm [6]. This mirror plane equation establishes the geometric relationship between real objects and their virtual counterparts.

## 4.3 Mirror Symmetry Mapping

Mirror symmetry is key to linking real 3D Gaussians with their virtual counterparts, enabling collaborative reconstruction. Symmetry mapping transforms 3D Gaussians across the mirror plane

<!-- image-->  
Figure 4: The Illustration of Symmetry Consistency Loss Function. In our method, we reflect the part directly captured $( G _ { r e a l }$ to $G _ { r e a l } ^ { \prime } )$ , and merge it with the mirror part $\left( G _ { m i r r o r } \right)$ using mirror region images to supervise training. We introduce a symmetry consistency loss, which measures the similarity between the $G _ { r e a l }$ and $G _ { r e a l } ^ { \prime }$ to enable collaborative optimization. Since the number of Gaussians changes during training, we also symmetrically transform the $G _ { m i r r o r } \mathrm { ~ }$ to $G _ { m i r r o r } ^ { \prime } ,$ and use the sceneâs overall symmetry to compare the center positions of the original Gaussians, computing the symmetry consistency loss.

to align virtual counterparts with real objects. The transformation relies on the mirror plane equation $ a x + b y + c z + d = 0$ , expressed as:

$$
\mathrm { S y m m e t r y } = \left[ { \begin{array} { c c c c } { 1 - 2 a ^ { 2 } } & { - 2 a b } & { - 2 a c } & { - 2 a d } \\ { - 2 a b } & { 1 - 2 b ^ { 2 } } & { - 2 b c } & { - 2 b d } \\ { - 2 a c } & { - 2 b c } & { 1 - 2 c ^ { 2 } } & { - 2 c d } \\ { 0 } & { 0 } & { 0 } & { 1 } \end{array} } \right] .\tag{5}
$$

For a Gaussian centered at $( x , y , z )$ , the mirrored coordinates $( x ^ { \prime } , y ^ { \prime } , z ^ { \prime } )$ are derived as:

$$
( x ^ { \prime } , y ^ { \prime } , z ^ { \prime } , 1 ) ^ { T } = \mathrm { S y m m e t r y } \cdot ( x , y , z , 1 ) ^ { T } .\tag{6}
$$

Additionally, the symmetry of the rotation and view-dependent color attributes is adjusted to align the real and virtual components. Symmetrized Gaussians are then integrated to achieve a unified representation, providing complementary information for reconstruction.

## 4.4 Symmetry-Aware Joint Optimization

To fully utilize mirror reflections, we introduce a spatial symmetry consistency loss that optimizes the integration of real and mirrored components. This loss measures the similarity between corresponding 3D Gaussians before and after symmetry transformation. As Shown in Figure 4, the Gaussiansâ centers are compared using normalized mean coordinate differences. However, due to dynamic changes in the number of Gaussians during training, mirrored Gaussians are also reflected back to the front of the mirror for comparison. The symmetry consistency loss is formulated as:

$$
L _ { \mathrm { s y m } } = \frac { 1 } { N } \sum \| \mu _ { G } - \mu _ { G } ^ { \prime } \| ,\tag{7}
$$

<!-- image-->  
Figure 5: Visualization of ReflectiveGS vs. Existing Methods Across Multiple Scenes in MirrorScene3D. Column 1: Cup. Column 2: Bag. Column 3: Duck. Column 4: Book. Column 5: Fruit. From the visual results, our method yields rendering outcomes with clearer and richer details, demonstrating superior rendering quality.

where $\mu _ { G } = G _ { r e a l } + G _ { r e a l } ^ { \prime } + G _ { m i r r o r } + G _ { m i r r o r } ^ { \prime }$ and $\mu _ { G } ^ { \prime }$ represent the centers of flip Gaussians of ÂµG, and N is the total number of Gaussians in the scene. By minimizing this loss, the training process ensures alignment and consistent optimization of real and mirrored components, enhancing the overall reconstruction quality.

Based on the previous content, the composition of our complete loss function is as follows:

$$
{ \cal L } = { \cal L } _ { r g b } + \lambda _ { m } { \cal L } _ { m } + \lambda _ { s y m } { \cal L } _ { s y m } ,\tag{8}
$$

where $L _ { r g b }$ is composed of the $L _ { 1 }$ and D-SSIM loss, which supervises the learning of various attributes of the Gaussians. $L _ { m }$ supervises the learning of the mirror factor, and $L _ { s y m }$ supervises the collaborative training of the specular front-and-back Gaussians. In our experiments, the hyperparameters are set to $\lambda _ { m } = 1 . 0$ and $\lambda _ { s y m } = 1 0 . 0$

## 5 Experiments

## 5.1 Experimental Setup

Metrics. We evaluate the performance using multiple metrics, including SSIM (Structural Similarity Index), PSNR (Peak Signal-to-Noise Ratio), LPIPS (Learned Perceptual Image Patch Similarity), and training time. These metrics collectively allow us to assess the similarity between rendered images and ground truth, providing a comprehensive measure of both visual fidelity and computational efficiency. To conduct a comprehensive comparison, we perform a quantitative evaluation by resizing all images to a uniform resolution of 1280Ã720 across all methods.

Implementation. We performed experiments across all scenes in our dataset, evaluating Vanilla 3DGS, Mirror-3DGS, and our proposed ReflectiveGS method. The aforementioned metrics were recorded for each approach to allow for a detailed comparison. All experiments were executed on a system equipped with a GeForce RTX 3090 GPU, ensuring consistent hardware conditions for evaluating the methods.

<table><tr><td>Scene</td><td colspan="4"> $C u p$ </td><td colspan="4">Bag</td><td colspan="4">Duck</td><td colspan="4">Book</td><td colspan="4">Fruit</td></tr><tr><td>Metrics</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td> $\begin{array} { l } { \mathrm { A v g . } } \\ { \mathrm { T i m e } } \end{array}$ </td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td> $\begin{array} { l } { \mathrm { A v g . } } \\ { \mathrm { T i m e } } \end{array}$ </td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>Avg. Time</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td> $\begin{array} { l } { \mathrm { A v g . } } \\ { \mathrm { T i m e } } \end{array}$ </td><td>SSIMâ</td><td>PSNRâ LPIPSâ</td><td></td><td>Avg. Time</td></tr><tr><td>Vanilla 3DGS</td><td>0.65</td><td>17.52</td><td>0.33</td><td>22m</td><td>0.59</td><td>15.07</td><td>0.42</td><td>12m</td><td>0.68</td><td>18.76</td><td>0.26</td><td>29m</td><td>0.65</td><td>13.92</td><td>0.44</td><td>13m</td><td>0.86</td><td>18.58</td><td></td><td>18m</td></tr><tr><td>Mirror-3DGS</td><td>0.49</td><td>15.44</td><td>0.47</td><td>53m</td><td>0.74</td><td>15.97</td><td>0.36</td><td>43m</td><td>0.72</td><td>19.77</td><td>0.41</td><td>53m</td><td>0.45</td><td>10.70</td><td>0.54</td><td>41m</td><td>0.87</td><td>18.30</td><td>0.22 0.18</td><td>47m</td></tr><tr><td>Ours</td><td>0.78</td><td>18.26</td><td>0.38</td><td>53m</td><td>0.85</td><td>17.01</td><td>0.26</td><td>43m</td><td>0.76</td><td>20.07</td><td>0.33</td><td>58m</td><td>0.90</td><td>16.88</td><td>0.27</td><td>41m</td><td>0.90</td><td>19.57</td><td>0.15</td><td>55m</td></tr></table>

Table 2: Comparison of Performance. We conducted experiments on the five scenes in our constructed dataset, evaluating Vanilla 3DGS, Mirror-3DGS, and ReflectiveGS (ours). The results were recorded for SSIM, PSNR, LPIPS, and training time, with the best values highlighted in bold. Our method achieves 23.8% higher SSIM, 1.59 dB higher PSNR, and 13.3% lower LPIPS compared to Vanilla 3DGS, with even greater improvements over Mirror-3DGS.

## 5.2 Experimental Results

We conducted comprehensive experiments to evaluate our method, with visual results shown in Figure 5. The following results demonstrate its advantages over baseline methods from multiple perspectives in mirror-augmented 3D reconstruction.

Mirror Estimation Error Analysis. Our method estimates mirror regions using learnable mirror factors, which are updated by comparing predicted masks with ground truth. While this effectively fits the mirror equation, some estimation errors persist. Experiments found that small errors have minimal impact on the results, and the resulting symmetry inaccuracies are optimized during the collaborative training process. When the mirror parameter error exceeds 5%, noticeable ghosting of objects appears in the rendered images.

Impact of Object Structure. ReflectiveGS performs well on simple objects, effectively integrating reflected information with real object. However, its performance slightly declines on complex one. In Figure 6, the geometry of objects affects the supplementation of mirrors. More complex surfaces require richer reflection data and higher precision in the mirror equation and symmetry consistency. However, our method maintains high rendering quality in general.

Optimization and Training Efficiency. Vanilla 3DGS had the shortest training time (around 20 minutes), but its performance is limited in complex mirror-containing scenes. Although ReflectiveGSâs training time is longer (around 50 minutes), the improvements in mirror information utilization and absent detail supplementation justify the additional computational cost. Compared to baseline methods, our approach achieves superior rendering results while maintaining competitive training times.

Rendering Quality Analysis. We quantitatively evaluated rendering quality using SSIM, PSNR, and LPIPS (see Table 2). The results show that our method consistently outperforms the baseline methods in SSIM and PSNR, demonstrating its superiority in detail and accurate reconstruction. And it achieves the lowest LPIPS values in most scenes, indicating that it provides the most visually realistic results. From the analysis of specific scenes, our method shows the most significant improvement in the Cup and Book scenes, demonstrating its strength in reconstructing regular objects. It also performs well in the Bag and Fruit scenes, highlighting its advantage in detail supplementation. Additionally, in the Duck scene, our method achieved the best PSNR, proving its precision in color and brightness reconstruction.

<!-- image-->  
Figure 6: Impact of Object Structure (âFruitâ Scene Example). In this scene, simpler structures (e.g., orange) have clearer rendering results, while more complex structures (e.g., lettuce) show more blurred visual effects.

<!-- image-->  
Figure 7: Comparison of Full Scene (Row 1) and Detail View (Row 2)(âFruitâ Scene Example). Our method utilizes mirror reflection information to enhance the 3D reconstruction, with the improvement being especially noticeable in the detail areas.

<table><tr><td rowspan="2">Metrics</td><td colspan="3">Full Scene</td><td colspan="3">Detail View</td></tr><tr><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td></tr><tr><td>Vanilla 3DGS</td><td>0.84</td><td>16.79</td><td>0.29</td><td>0.74</td><td>15.1</td><td>0.54</td></tr><tr><td>Mirror-3DGS</td><td>0.87</td><td>16.61</td><td>0.21</td><td>0.87</td><td>16.51</td><td>0.45</td></tr><tr><td>ReflectiveGS</td><td>0.90</td><td>17.67</td><td>0.18</td><td>0.91</td><td>17.72</td><td>0.36</td></tr></table>

Table 3: Metrics of Full Scene and Detail View (âFruitâ Scene Example). Our method outperforms the baseline in all metrics, with the improvement being more pronounced in the detail view compared to the full scene, highlighting our methodâs advantage in supplementing occluded details.

Detail Performance. To better assess the detail reconstruction, we conducted the same quantitative comparison using SSIM, PSNR and LPIPS for the occluded regions defined in Section 3. The visual results (Figure 7) and metrics (Table 3) show a more significant improvement in detail view than full scene, with a higher improvement rate across all metrics relative to the baseline. This demonstrates the effectiveness of our method in utilizing mirror information for 3D reconstruction, particularly in supplementing details in viewpoint blind spots.

Summary. Experimental results show that ReflectiveGS consistently outperforms baseline methods. Compared to Vanilla 3DGS, it achieves an average improvement of 23.8% in SSIM, 1.59 dB in PSNR, and a 13.3% reduction in LPIPS, with even greater improvements over Mirror-3DGS, averaging 36.6% in SSIM, 2.32 dB in PSNR, and a 26.6% decrease in LPIPS. Despite challenges like object structure and mirror equation errors, it remains overall more effective. Using mirror reflections, it enhances reconstruction accuracy, particularly in detail supplementation, while maintaining high-quality rendering with competitive training speed.

## 5.3 Ablation Study

We perform ablation experiments to evaluate the impact of key components: Mirror Information SupplementatioCn, Mirror Equation Estimation, and Symmetry Consistency Loss. Table 4 summarizes the quantitative results, and Figure 8 presents the visual comparisons.

<!-- image-->  
Figure 8: Ablation Study Visualization (âBagâ Scene Example). The figure presents fullscene and detail view renderings across three ablation experiments. Removing mirror reflection loses supplementary details, mirror equation errors cause blurring and ghosting, and omitting symmetry consistency loss results in inaccurate renderings and unclear boundaries. These components collectively enhance reconstruction quality.

<table><tr><td>Scene</td><td colspan="3">Cup</td><td colspan="3">Bag</td><td colspan="3">Duck</td><td colspan="3">Book</td><td colspan="3">Fruit</td><td colspan="3">Average</td></tr><tr><td>Metrics</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>| SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>|SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td></tr><tr><td>No-MirrorReflection</td><td></td><td>17.83</td><td>0.33</td><td>0.81</td><td>18.47</td><td>0.23</td><td>0.70</td><td>19.07</td><td>0.32</td><td>0.46</td><td>13.11</td><td>0.49</td><td>0.79</td><td>17.27</td><td>0.24</td><td>0.68</td><td>17.15</td><td>0.32</td></tr><tr><td>MirrorEstimationError(5%)</td><td>0.63 0.68</td><td>17.27</td><td>0.40</td><td>0.85</td><td>18.28</td><td>0.23</td><td>0.77</td><td>19.76</td><td>0.35</td><td>0.84</td><td>15.05</td><td>0.36</td><td>0.68</td><td>13.43</td><td>0.32</td><td>0.76</td><td>16.76</td><td>0.33</td></tr><tr><td>No-SymmetryConsistencyLoss</td><td>0.73</td><td>18.26</td><td>0.42</td><td>0.87</td><td>18.00</td><td>0.20</td><td>0.77</td><td>20.08</td><td>0.41</td><td>0.89</td><td>16.27</td><td>0.32</td><td>0.88</td><td>17.92</td><td>0.19</td><td>0.83</td><td>18.11</td><td>0.31</td></tr><tr><td>FulFl-Model</td><td>0.82</td><td>19.49</td><td>0.37</td><td>0.90</td><td>18.27</td><td>0.19</td><td>0.78</td><td>20.15</td><td>0.34</td><td>0.91</td><td>17.11</td><td>0.28</td><td>0.90</td><td>19.05</td><td>0.18</td><td>0.86</td><td>18.81</td><td>0.27</td></tr></table>

Table 4: Ablation Study Results for SSIM, PSNR, and LPIPS. In the ablation study, we tested the contributions of mirror information supplementation and the symmetry consistency loss function to the overall method. The complete model with all components consistently achieved the best performance. The best values highlighted in bold.

Mirror Information Supplementation. A white overlay mask was applied to the mirror regions to evaluate the impact of reflection information on reconstruction quality. The results show that covering the mirror reflections leads to a loss of supplementary information, preventing the model from recovering details, particularly those from occluded or viewpoint-blind areas. This highlights the critical role of mirror reflections in enhancing reconstruction quality.

Mirror Equation Estimation. An error term was introduced to the mirror equation, causing a deviation 5% from the parameters originally fitted. The results show visible ghosting in rendered images and degraded metrics, demonstrating precise mirror equation estimation is essential for maintaining symmetry between real and virtual Gaussians, forming a foundation for symmetry and joint optimization.

Symmetry Consistency Loss. The symmetry consistency loss was removed by setting its hyperparameter to 0. This led to misalignment between real and mirrored Gaussians, causing unclear boundaries and inaccuracy rendering. The results confirm the necessity of symmetry constraints in preserving structural coherence and improving reconstruction accuracy.

## 6 Conclusion

In this work, We introduced ReflectiveGS, a novel approach that integrates mirror reflections into 3D scene reconstruction. Unlike existing methods that focus on rendering mirror surfaces, our method leverages mirror reflections to recover absent viewpoints, enhancing reconstruction accuracy and detail preservation. To support this task, we constructed MirrorScene3D, a dedicated dataset for mirror-augmented reconstruction, providing a benchmark for evaluating the effectiveness of mirror reflections as supplementary information. Experiments show that ReflectiveGS outperforms Vanilla 3DGS and Mirror-3DGS in rendering quality, achieving higher SSIM and PSNR with lower

LPIPS while maintaining competitive training efficiency.

Despite these advancements, certain challenges remain. Mirror estimation errors can occur in complex or weakly reflective regions, affecting the modelâs performance and requiring further optimization. Additionally, scalability to large-scale real scenes requires further investigation. Future work will focus on improving mirror estimation robustness, optimizing computational efficiency, and extending the approach to broader scene categories to enhance its real-world applicability.

## References

[1] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5855â5864, 2021.

[2] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased grid-based neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19697â19705, 2023.

[3] Jiazhong Cen, Jiemin Fang, Chen Yang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian. Segment any 3d gaussians. arXiv preprint arXiv:2312.00860, 2023.

[4] Eric R Chan, Koki Nagano, Matthew A Chan, Alexander W Bergman, Jeong Joon Park, Axel Levy, Miika Aittala, Shalini De Mello, Tero Karras, and Gordon Wetzstein. Generative novel view synthesis with 3d-aware diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4217â4229, 2023.

[5] Kangle Deng, Andrew Liu, Jun-Yan Zhu, and Deva Ramanan. Depth-supervised nerf: Fewer views and faster training for free. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12882â12891, 2022.

[6] Martin A Fischler and Robert C Bolles. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM, 24(6):381â395, 1981.

[7] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5501â5510, 2022.

[8] Leif Van Holland, Ruben Bliersbach, Jan U MÃ¼ller, Patrick Stotko, and Reinhard Klein. Tramnerf: Tracing mirror and near-perfect specular reflections through neural radiance fields. In Computer Graphics Forum, volume 43, page e15163. Wiley Online Library, 2024.

[9] Sheng Hong, Junjie He, Xinhu Zheng, Chunran Zheng, and Shaojie Shen. Liv-gaussmap: Lidar-inertial-visual fusion for real-time 3d radiance field map rendering. IEEE Robotics and Automation Letters, 2024.

[10] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 conference papers, pages 1â11, 2024.

[11] Jiajun Huang, Hongchuan Yu, Jianjun Zhang, and Hammadi Nait-Charif. Pointân move: Interactive scene object manipulation on gaussian splatting radiance fields. IET Image Processing, 18(12):3507â3517, 2024.

[12] Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xiaoxiao Long, Wenping Wang, and Yuexin Ma. Gaussianshader: 3d gaussian splatting with shading functions for reflective surfaces. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5322â5332, 2024.

[13] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, and Jonathon Luiten. Splatam: Splat track & map 3d gaussians for dense rgb-d slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21357â21366, 2024.

[14] Bernhard Kerbl, Georgios Kopanas, Thomas LeimkÃ¼hler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023.

[15] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park. Compact 3d gaussian representation for radiance field. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21719â21728, 2024.

[16] Zhihao Liang, Qi Zhang, Wenbo Hu, Lei Zhu, Ying Feng, and Kui Jia. Analytic-splatting: Anti-aliased 3d gaussian splatting via analytic integration. In European conference on computer vision, pages 281â297. Springer, 2024.

[17] Jiaying Lin, Guodong Wang, and Rynson WH Lau. Progressive mirror detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3697â3705, 2020.

[18] Jiayue Liu, Xiao Tang, Freeman Cheng, Roy Yang, Zhihao Li, Jianzhuang Liu, Yi Huang, Jiaqi Lin, Shiyong Liu, Xiaofei Wu, et al. Mirrorgaussian: Reflecting 3d gaussians for reconstructing mirror reflections. arXiv preprint arXiv:2405.11921, 2024.

[19] Yuan Liu, Peng Wang, Cheng Lin, Xiaoxiao Long, Jiepeng Wang, Lingjie Liu, Taku Komura, and Wenping Wang. Nero: Neural geometry and brdf reconstruction of reflective objects from multiview images. ACM Transactions on Graphics (ToG), 42(4):1â22, 2023.

[20] Ricardo Martin-Brualla, Noha Radwan, Mehdi SM Sajjadi, Jonathan T Barron, Alexey Dosovitskiy, and Daniel Duckworth. Nerf in the wild: Neural radiance fields for unconstrained photo collections. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7210â7219, 2021.

[21] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and Andrew J Davison. Gaussian splatting slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18039â18048, 2024.

[22] Jiarui Meng, Haijie Li, Yanmin Wu, Qiankun Gao, Shuzhou Yang, Jian Zhang, and Siwei Ma. Mirror-3dgs: Incorporating mirror reflections into 3d gaussian splatting. arXiv preprint arXiv:2404.01168, 2024.

[23] Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics (ToG), 38(4):1â14, 2019.

[24] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021.

[25] Thomas MÃ¼ller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):1â15, 2022.

[26] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10318â10327, 2021.

[27] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104â4113, 2016.

[28] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-Hua Guan. Language embedded 3d gaussians for open-vocabulary scene understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5333â5343, 2024.

[29] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653, 2023.

[30] Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zickler, Jonathan T Barron, and Pratul P Srinivasan. Ref-nerf: Structured view-dependent appearance for neural radiance fields. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5481â5490. IEEE, 2022.

[31] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. arXiv preprint arXiv:2106.10689, 2021.

[32] Olivia Wiles, Georgia Gkioxari, Richard Szeliski, and Justin Johnson. Synsin: End-to-end view synthesis from a single image. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7467â7477, 2020.

[33] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20310â20320, 2024.

[34] Tong Wu, Yu-Jie Yuan, Ling-Xiao Zhang, Jie Yang, Yan-Pei Cao, Ling-Qi Yan, and Lin Gao. Recent advances in 3d gaussian splatting. Computational Visual Media, pages 1â30, 2024.

[35] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong Wang, and Xuelong Li. Gs-slam: Dense visual slam with 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19595â19604, 2024.

[36] Zhiwen Yan, Weng Fei Low, Yu Chen, and Gim Hee Lee. Multi-scale 3d gaussian splatting for anti-aliased rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20923â20931, 2024.

[37] Xin Yang, Haiyang Mei, Ke Xu, Xiaopeng Wei, Baocai Yin, and Rynson WH Lau. Where is my mirror? In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8809â8818, 2019.

[38] Ziyi Yang, Xinyu Gao, Yang-Tian Sun, Yihua Huang, Xiaoyang Lyu, Wen Zhou, Shaohui Jiao, Xiaojuan Qi, and Xiaogang Jin. Spec-gaussian: Anisotropic view-dependent appearance for 3d gaussian splatting. Advances in Neural Information Processing Systems, 37:61192â61216, 2025.

[39] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20331â20341, 2024.

[40] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke. Gaussian grouping: Segment and edit anything in 3d scenes. arXiv preprint arXiv:2312.00732, 2023.

[41] Ze-Xin Yin, Jiaxiong Qiu, Ming-Ming Cheng, and Bo Ren. Multi-space neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12407â12416, 2023.

[42] Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and Angjoo Kanazawa. Plenoctrees for real-time rendering of neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5752â5761, 2021.

[43] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4578â4587, 2021.

[44] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19447â19456, 2024.

[45] Vladimir Yugay, Yue Li, Theo Gevers, and Martin R Oswald. Gaussian-slam: Photo-realistic dense slam with gaussian splatting. arXiv preprint arXiv:2312.10070, 2023.

[46] Junyi Zeng, Chong Bao, Rui Chen, Zilong Dong, Guofeng Zhang, Hujun Bao, and Zhaopeng Cui. Mirror-nerf: Learning neural radiance fields for mirrors with whitted-style ray tracing. In Proceedings of the 31st ACM International Conference on Multimedia, pages 4606â4615, 2023.

[47] Kai Zhang, Gernot Riegler, Noah Snavely, and Vladlen Koltun. Nerf++: Analyzing and improving neural radiance fields. arXiv preprint arXiv:2010.07492, 2020.

[48] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21676â21685, 2024.

[49] Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and Markus Gross. Ewa volume splatting. In Proceedings Visualization, 2001. VISâ01., pages 29â538. IEEE, 2001.