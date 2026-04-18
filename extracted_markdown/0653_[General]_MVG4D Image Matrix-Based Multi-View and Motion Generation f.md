# MVG4D: Image Matrix-Based Multi-View and Motion Generation for 4D Content Creation from a Single Image

Dongfu Yin1, Xiaotian Chenâ1,2, Fei Richard Yu1, Xuanchen Li3, and Xinhao Zhang4

1Guangdong Laboratory of Artificial Intelligence and Digital Economy (Shenzhen)

2Shenzhen University

3Tsinghua University

4Dalian University of Technology

## Abstract

Advances in generative modeling have significantly enhanced digital content creation, extending from 2D images to complex 3D and 4D scenes. Despite substantial progress, producing high-fidelity and temporally consistent dynamic 4D content remains a challenge. In this paper, we propose MVG4D, a novel framework that generates dynamic 4D content from a single still image by combining multi-view synthesis with 4D Gaussian Splatting (4D GS). At its core, MVG4D employs an image matrix module that synthesizes temporally coherent and spatially diverse multi-view images, providing rich supervisory signals for downstream 3D and 4D reconstruction. These multi-view images are used to optimize a 3D Gaussian point cloud, which is further extended into the temporal domain via a lightweight deformation network. Our method effectively enhances temporal consistency, geometric fidelity, and visual realism, addressing key challenges in motion discontinuity and background degradation that affect prior 4D GS-based methods. Extensive experiments on the Objaverse dataset demonstrate that MVG4D outperforms state-of-the-art baselines in CLIP-I, PSNR, FVD, and time efficiency. Notably, it reduces flickering artifacts and sharpens structural details across views and time, enabling more immersive AR/VR experiences. MVG4D sets a new direction for efficient and controllable 4D generation from minimal inputs.

Keywords: Single image to 4D content, 4D Gaussian Splatting (4D GS), Multi-view image generation, Dynamic scene reconstruction

## 1 Introduction

With the rapid development of generative modeling technologies, the ability to create various digital contents such as 2D images [26], videos [1, 3, 7], and 3D scenes [21, 17, 4, 11] has been significantly improved. However, generating continuous and high-quality dynamic 4D scenes [28, 16] remains challenging. Dynamic 4D scenes are typically represented by Dynamic Neural Radiance Fields(NeRF) [23],which aim to model appearance, geometry, and motion across different viewpoints. For instance, Hexplane [2] technology enables the generation of dynamic videos directly from text, which can then be converted into 4D presentations through advanced diffusion models, greatly enriching content creation possibilities. Emerging technologies such as CONSISTENT4D [13] and MAV3D [28], attempt to generate coherent 4D scenes from a single video source by optimizing the NeRF framework and diffusion models. These methods effectively combine video information with dynamic 3D models to generate 4D content rich in details and dynamic textures. Additionally, techniques such as Animate124 [38] explore the transformation of a single image into a dynamic 3D video, further expanding the applications of dynamic scene generation.

Despite the potential of these advanced techniques, challenges remain in terms of clarity and time efficiency. Current methods may require long processing times to generate high-quality dynamic NeRF, and the sharpness of the generated dynamic scenes needs improvement. Enhancing efficiency and clarity while ensuring quality remains a key focus of current research in 4D generation technology.

Recent works have introduced time-set variables to optimize Gaussian deformation fields based on the 3D Gaussian Splatting (3D GS) [14] technique, resulting in a novel 4D representation called 4D Gaussian Splatting (4D GS) [34]. This method aims to achieve efficient training and storage for real-time dynamic scene rendering while maintaining high-quality output. The 4D GS method employs a lightweight MLP to predict 3D Gaussian deformations for new timestamps, offering a novel display representation for dynamic scenes. This approach significantly reduces optimization time and enables real-time processing while maintaining high-quality deformable Gaussian Splatting.

While 4D Gaussian Splatting (4D GS) offers an efficient method for 4D scene representation and has significantly improved optimization time for 4D content, challenges remain in generating 4D content from a single image using 4D GS technology. These challenges include issues related to the absence of background points and inaccuracies in camera positioning, all of which impede the precise optimization of dynamic scenes. Additionally, during motion, the displacement of Gaussian points can lead to surface tearing on 3D objects, compromising the quality and realism of the rendering. Therefore, the current methods for real-time 4D scene generation based on 4D GS still require further optimization. To address these challenges, we propose a multi-view image generation model that captures comprehensive multi-view information of the input object, addressing issues like missing background points and imprecise camera positioning while significantly reducing the generation time for 4D scenes.

Our work can be summarized as follows:

1. We introduce an image matrix generation module that produces diverse and temporally coherent multi-view images from a single input image. This module provides rich spatial and temporal supervision, significantly enhancing the clarity, consistency, and realism of 4D dynamic content.

2. We develop a compact 3D scene representation by initializing and optimizing 3D Gaussian point clouds directly from the generated image matrix. This representation enables accurate reconstruction with reduced computational overhead, serving as a strong foundation for dynamic modeling.

3. We build an integrated framework that transforms a single image into dynamic 4D content by sequentially combining multi-view generation, 3D reconstruction, and 4D optimization. The proposed method achieves high visual fidelity and temporal consistency while significantly reducing generation time.

## 2 Related Work

## 2.1 Single View to Multi-view Image Generation

Exciting progress has been made in using diffusion models to generate images from multiple views using a single image of an object. For example, Zero123 [18] utilizes the geometric prior learned by a large-scale diffusion model to train a conditional diffusion model on synthetic datasets, achieving zero-shot synthesis of new perspective images from single-view inputs. By combining images from different perspectives as target images, Zero123++ [27] significantly overcomes the problem of no correlation between the images generated by Zero123, and improves the quality and consistency of generating consistent multi-perspective images from a single image. SyncDreamer [20] further optimizes its adaptability to downstream 3D generation tasks on the basis of the previous two, but it still has shortcomings due to the high hardware requirements.

Although significant progress has been made in generating multi-view images from a single input, there are still several drawbacks. The main problem is that it is difficult to maintain the geometric consistency of objects when generating images from other viewpoints. The generated images may exhibit structural and texture inconsistencies. In addition, the problem of possible loss of details is also a key problem to be solved urgently, especially in images with background. Our method addresses these issues by employing a fine-tuned multi-view image generation module. This approach enhances the continuity and clarity of the generated image by minimizing the viewpoint rotation. This method effectively improves the geometric consistency of the objects in the generated images while reducing the loss of details.

## 2.2 Image-to-3D Generation

The technology for generating 3D models from images has been rapidly maturing, and many 3D models representation methods based on point clouds [4, 11], meshes, implicit neural representations [24, 5] and Gaussian Splatting have emerged. Most of these methods, which are built from multi-view images, focus on creating a 3D representation of the scene. One-2-3-45++ [17] combines a 2D diffusion model and a 3D native diffusion model for multi-view conditions to quickly generate 3D meshes through consistent multi-view image generation and 3D reconstruction. However, this traditional diffusion model-based scheme generally has the problem of long optimization time. 3D GS [14] uses 3D Gaussian scene representation and real-time differentiable renderer to achieve real-time rendering of radiation field, which effectively reduces the time required for 3D content reconstruction and achieves high-quality scene representation by optimizing 3D Gaussian attributes and density control. While the task of real-time 3D reconstruction with just one image input has not yet been addressed, LRM [12] uses a Transformer [31]-based encoder-decoder architecture to predict Neural Radiance Fields (NeRF) directly from a single image for 3D reconstruction. This has significant implications for the task of 3D reconstruction based on a single image.

Current methods for 3D content generation often rely on iterative optimization of the target model using diffusion models. However, this approach can be time-consuming, leading to high computational costs. In contrast, Transformer-based encoder-decoder architectures offer better scalability and efficiency but can produce blurred textures in occluded areas, resulting in visual distortion. Recent innovations using 3D Gaussian Splatting (3D GS) for 3D content representation excel in real-time rendering. Nevertheless, these methods may exhibit artifacts in regions not observed during training. Our approach addresses these challenges by incorporating a multi-view image generation module. This module generates additional views of the input image, thereby reducing artifacts caused by occluded regions. Additionally, we use 3D GS to represent the 3D scene, which effectively reduces computational time and improves optimization.

## 2.3 4D Generation

Dynamic scene rendering, also known as 4D generation work, aims to enable real-time rendering of dynamic 3D scenes with efficient training and storage. In this field, neural radiance Fields (NeRF) techniques have been widely used to represent 4D scenes. For example, Neural-3D-Video [15] uses time-conditioned NeRF and a series of compact latent codes to represent dynamic scenes, and uses a hierarchical training scheme and light importance sampling to significantly improve the training speed and the perceptual quality of generated images. TiNeuVox [8] accelerates the optimization of dynamic radiance fields by combining time-aware voxel features and micro-coordinate deformation networks to represent scenes. This work significantly reduces training time and storage costs while maintaining high-quality rendering. However, the robustness of dynamic radiance field reconstruction methods remains a challenge. RoDynRFâs [19] method for reconstructing camera trajectories and dynamic radiance fields from randomly captured dynamic monocular videos, enabling rendering that focuses on robust dynamic scenes. And MSTH [33], which uses combinations of hash codes to represent dynamic scenes.

With the development of 3D Gaussian Splatting (3D GS), 4D Gaussian Splatting (4D GS) provides an innovative way to render dynamic scenes, which handles dynamic changes through continuous 4D representation. The 3D GS function is manipulated to adapt to temporal and spatial changes, thus enabling real-time rendering and high-resolution output. 4D GS emphasizes computation and storage efficiency, and utilizes a compact network structure to efficiently capture complex dynamic changes while maintaining high-quality rendering effects. Furthermore, DreamGaussian4D [25] uses an image-to-3D framework to fit a static 3D GS function, and then optimalizes the dynamic representation by learning the motion that drives the video. Finally, the 4D GS is exported as an animated mesh sequence, and the texture map is optimized through a video-to-video process. Recently, Stable Video 4D (SV4D) [35] provides an innovative approach for dynamic 3D content generation that enables 4D generation via multi-frame multi-view coherence. Unlike traditional approaches that rely on independently trained video generation and novel view synthesis models, SV4D employs a unified diffusion model that is able to generate multi-view videos of dynamic 3D objects from a mutil-view video.

Despite these improvements, issues like motion discontinuity and high computation persist. Our approach enhances dynamic rendering with improved image matrix module and efficient 4D GS optimization, yielding better quality and temporal continuity.

## 3 Method

The technique for generating 4D content from a single image provides an innovative approach to 4D content generation. Our proposed method, MVG4D, combines a image matrix module with the 4D Gaussian Splatting (4D GS) dynamic content optimization technique. This method is based on 3D Gaussian Splatting (3D GS) technology to optimize the Gaussian deformation field, which not only enhances the continuity and clarity of the generated 4D content but also significantly accelerates the 4D content generation process. Our study is illustrated in Figure 1, which outlines three main phases: Image matrix module, 3D Gaussian Splatting Construction, and 4D Content Synthesis Using 4D Gaussian Splatting.

<!-- image-->  
Figure 1: The overall pipeline of MVG4D. MVG4D is divided into three main stages: The green background part is the Image matrix module, the blue background part is the 3D Gaussian Splatting construction, and the red background part is the 4D content synthesis using 4D Gaussian Splatting.

## 3.1 Image matrix module

To construct a comprehensive and time-aware multi-view image matrix from a single input image $I _ { 0 } ,$ , we propose a two-stage processing flow that takes full advantage of the multi-view generation as well as the introduction of dynamic information.

Given the input image $I _ { 0 }$ , we first utilize the pre-trained module to synthesize a video that encapsulates dynamic information. This video is then decomposed into a set of video frames $I _ { t } ,$ where t denotes the timestamp. Although this video captures temporal dynamics, it contains only single-view information. To achieve multi-view dynamic content generation, we fine-tune a diffusion-based multi-view image generation model to synthesize novel viewpoints for each frame $I _ { t } .$ The resulting image matrix embeds both temporal and viewpoint diversity, serving as a supervisory signal for optimizing the subsequent 4D GS representation.

The core of our approach lies in fine-tuning a view-conditional 2D diffusion model to generate consistent multi-view images for each video frame. During inference, the model takes as input the original frame image and the desired relative camera parameters such as angular offsets and depth variationsâand outputs novel-view images accordingly. During training, we place the object at the origin of a canonical 3D coordinate system and simulate a spherical camera setup. The camera is positioned on a sphere centered at the object and is constrained to always face the origin. Let two camera viewpoints be defined by spherical coordinate $( \theta _ { 1 } , \phi _ { 1 } , r _ { 1 } )$ and $( \theta _ { 2 } , \phi _ { 2 } , r _ { 2 } )$ , representing the polar angle, azimuth angle, and radius, respectively. Their relative transformation is parameterized as $( \Delta \theta , \Delta \phi , \Delta r ) = ( \theta _ { 2 } - \theta _ { 1 } , \phi _ { 2 } - \phi _ { 1 } , r _ { 2 } - r _ { 1 } )$

The training objective of the diffusion model is to learn a function f such that given an input image x1 $x _ { 1 }$ and a viewpoint transformation $( \Delta \theta , \Delta \phi , \Delta r )$ , the model can generate a novel-view image that closely resembles the ground truth image $x _ { 2 }$ , which is captured from the target viewpoint. This process is illustrated in Figure 2. The model thus learns a general mechanism for camera viewpoint control and can infer the target image $x _ { 2 }$ from $x _ { 1 }$ under arbitrary relative viewpoint changes.

The training objective is expressed as:

$$
\operatorname* { m i n } _ { \theta } \mathbb { E } _ { z \sim \mathcal { E } ( x ) , t , \epsilon \sim N ( 0 , 1 ) } \left. \epsilon - \epsilon _ { \theta } ( z _ { t } , t , c ( x , R , T ) ) \right. _ { 2 } ^ { 2 }\tag{1}
$$

In this formula, x denotes the input image, $c ( x , R , T )$ represents the conditioning embedding that incorporates the input image and the target viewpoint information, t is the diffusion timestep, E is the image encoder, ÏµÎ¸ is a U-Net-based denoiser, and $z _ { t }$ is the latent representation of x at timestep t.

<!-- image-->  
Figure 2: The learning process of the multi-view image generation model. The input image, along with the target view information, is provided as input to a fine-tuned diffusion model, which then generates the corresponding target image.

Due to the use of different relative view parameters for training, the novel-view images generated from each frame $I _ { t }$ may exhibit inconsistent viewpoint alignments. This inconsistency introduces mismatches in the constructed image matrix, adversely affecting the quality of 4D content generation. To resolve this issue, we further fine-tune the diffusion model using carefully aligned supervision to ensure consistency across views and time. Specifically, during training, we minimize perceptual discrepancies between the generated novel-view images and the corresponding input frames $I _ { t } ,$ enforcing visual alignment between the synthesized views and the original inputs. This enhancement enables us to generate a spatiotemporally coherent image matrix, thereby improving the downstream optimization of 4D GS.

## 3.2 3D Gaussian Splatting Construction

After obtaining continuous and high-quality multi-view images, we utilize advanced 3D reconstruction technology to transform these 2D images into a 3D GS model. This process consists of two key steps.

## 3.2.1 Multi-view Fusion

We employ multi-view fusion technology to integrate image information captured from various perspectives. This technique leverages the geometric and illumination consistency between images to accurately reconstruct the 3D structure.

## 3.2.2 Volumetric Reconstruction

We initialize unity-scaled, rotation-free 3D Gaussian point clouds at random positions within the space periodically densifying them during optimization. Unlike the reconstruction pipeline, we begin with fewer Gaussian point clouds but densify them more frequently to align with the generation process. We optimize the 3D Gaussian point clouds using Score Distillation Sampling (SDS) loss. At each step, we sample random camera poses centered around the object and render the RGB image from the current viewpoint. During training, we linearly decrease the time step $t ,$ which weights the random noise added to the rendered RGB image. The input image then serves as a 2D diffusion prior, which is used to optimize the underlying 3D Gaussian point clouds with the SDS loss:

$$
\nabla _ { \Theta } \mathcal { L } _ { S D S } = \mathbb { E } _ { t , p , \epsilon } \left[ \omega _ { \left( t \right) } \left( \epsilon _ { \phi } ( I ; t , I ^ { r } , \triangle p ) - \epsilon \right) \frac { \partial I } { \partial \Theta } \right]\tag{2}
$$

Where $\omega _ { ( t ) }$ is the weighting function, $\epsilon _ { \phi }$ represents the noise predicted by the 2D diffusion prior $\phi , \ \triangle p$ is the relative change in camera pose with respect to the reference camera r. I is the input image, and $I ^ { r }$ is the image of the new pose obtained by the 2D diffusion model. Through this method, we can effectively recover the 3D geometry and texture information from 2D images, providing a solid foundation for the subsequent generation of 4D dynamic models. The optimization goal of this stage is to enhance the geometric accuracy and visual-realism of the 3D model. Additionally, it aims to ensure the modelâs consistency when viewed from various angles.

## 3.3 4D Content Synthesis Using 4D Gaussian Splatting

After obtaining the image matrix containing multi-view dynamic information of the input image and the initialized 3D GS model, we convert the static 3D GS point cloud into a dynamic 4D model. We extract the central coordinates $( x , y , z )$ and timestamp t of each 3D Gaussian point cloud. A Spatial-Temporal Structure Encoder is then utilized to calculate the characteristics of the voxels. These features are analyzed and decoded using a micro-MLP to obtain the deformed 3D Gaussian point cloud at timestamp $t ,$ render the deformed point cloud image, and match the image matrix generated in the first stage. The mean squared error (MSE) loss is calculated between the image matrix generated in the first stage and the images rendered by the deformed point cloud:

$$
\mathcal { L } _ { \mathrm { R e f } } = \frac { 1 } { \tau } \sum _ { \tau = 1 } ^ { \tau } \Vert f ( \phi ( S , \tau ) , o _ { \mathrm { R e f } } ) - I _ { \mathrm { R e f } } ^ { \tau } \Vert _ { 2 } ^ { 2 }\tag{3}
$$

In this formula, $\tau$ is a time variable, $o _ { \mathrm { R e f } }$ denotes the viewpoint information corresponding to the image matrix generated in the first stage, with the associated image represented as $I _ { \mathrm { R e f } } ^ { \tau } .$ The 4D deformation field model $f$ is used to render the image from this viewpoint, and the error between the rendered image and $I _ { \mathrm { R e f } } ^ { \tau }$ is calculated to optimize the model $f .$ To comprehensively address the challenge of modeling occluded parts of the scene, we utilize a 3D-aware image diffusion model based on the multi-view images generated in the initial stage. This approach enables us to capture more complete and detailed scene information, which is then used to reverse-optimize the 4D Gaussian deformation field through SDS loss.

Through our research on the proposed method MVG4G, we can not only reconstruct accurate 3D models from a single image but also generate dynamic 4D content. Compared to traditional methods, our technology significantly improves generation efficiency, the naturalness of dynamic content, and visual continuity. The optimized model also demonstrates enhanced clarity and reduced motion flicker in dynamic performances. These improvements are crucial for the practical application of 4D content in fields such as augmented reality and virtual reality, particularly for creating more realistic and immersive experiences.

## 4 Experiment

## 4.1 Experiment Setup

## 4.1.1 Dataset

To evaluate the effectiveness of our method in generating high-quality dynamic 4D content, we conducted qualitative and qualitative evaluations based on datasets such as Objaverse [6].

## 4.1.2 Evaluation Metrics

For the task of generating 4D content from a single input image, designing appropriate evaluation metrics is essential to comprehensively assess the quality, consistency, and realism of the generated multi-view and dynamic content. Our method is evaluated using the following metrics: CLIP-I, PSNR, Time, FVD [30], each targeting different aspects of performance:

â¢ CLIP-I: Measures perceptual similarity between the generated image and the input image by computing their cosine similarity in the CLIP embedding space.

â¢ PSNR: Evaluates pixel-level reconstruction fidelity between the generated image and the input image; higher values indicate better low-level similarity.

â¢ Time: Reports the inference time required to generate the full 4D content from a single image.

â¢ FVD-F: calculate FVD over frames at each view.

â¢ FVD-Diag: calculate FVD over the diagonal images of the image matrix.

â¢ FV4D: calculate FVD over all images by scanning them in a bidirectional raster order.

## 4.1.3 Baselines

To evaluate the quality of dynamic 4D content generation from a single image, we compare our method MVG4D with several recent 4D content generation methods, including DreamGaussian4D [25], V4D [10], 4Diffusion [37], etc. We use official code published by the respective authors to generate comparison results.

## 4.2 Implementation Details

In all of our experiments, we only used a single NVIDIA RTX 4090 GPU. We process a single image of the input using a fine-tuned diffusion model to obtain image matrix supervision that optimizes 4D GS to generate dynamic 4D content.

Table 1: CLIP Image Similarity (CLIP-I). CLIP-I assesses the similarity between the input image and the generated 4D rendered image by computing their cosine similarity. The normalized similarity values range from 0 to 1, with higher values indicating that the generated 4D content is more similar to the input image. Compared with current mainstream and novel 4D scene generation methods, the 4D rendered images produced by MVG4D exhibit greater similarity to the input images, highlighting the advantages of our approach in 4D content generation.
<table><tr><td>Method</td><td>CLIP-I â</td></tr><tr><td>RealFusion-V [22]</td><td>0.803</td></tr><tr><td>Animate124 [38]</td><td>0.854</td></tr><tr><td>DreamGaussian4D [25]</td><td>0.923</td></tr><tr><td>Consistent4D [13]</td><td>0.921</td></tr><tr><td>EG4D [29]</td><td>0.954</td></tr><tr><td>MVG4D (ours)</td><td>0.982</td></tr></table>

## 4.3 Quantitative Results

## 4.3.1 Image Similarity Metrics

The experimental results, as shown in Table 1, demonstrate that the 4D rendered images generated by the MVG4D method exhibit significantly higher CLIP-I similarity to the input image compared to those produced by existing mainstream and novel 4D scene generation methods. This suggests that our approach has made substantial progress in enhancing image similarity. The improved performance is likely due to the use of multi-view images, which are generated by our multi-view image generation module. These images provide additional spatial information that effectively guides the optimization of the 4D GS method. This innovation enables us to generate more accurate 4D content compared to methods that rely on video frames.

Table 2: Peak Signal-to-Noise Ratio (PSNR). PSNR is used to assess the quality of 4D scenes generated from a single input image. Our method is benchmarked against several existing methods, showing a significant improvement in PSNR values. Compared to the baseline methods, our approach significantly reduces visual distortion in the reconstructed scene, achieving higher fidelity and more precise generation of dynamic 4D content.
<table><tr><td>Method</td><td>PSNR â</td></tr><tr><td>TiNeuVox-B [8]</td><td>32.67</td></tr><tr><td>DreamGaussian4D [25]</td><td>34.05</td></tr><tr><td>KPlanes [9]</td><td>31.61</td></tr><tr><td>V4D [10]</td><td>33.72</td></tr><tr><td>4Diffusion b [37]</td><td>35.07</td></tr><tr><td>MVG4D (ours)</td><td>36.44</td></tr></table>

As shown in Table 2, when compared with state-of-the-art 4D scene generation methods, the 4D rendering results generated by MVG4D achieved significantly higher PSNR scores, highlighting the effectiveness and advancement of the proposed method. This advantage is likely due to our use of 4D Gaussian point clouds to represent 4D scenes, which better accumulates spatial point information compared to traditional NeRF-based methods.

Table 3: Evaluation of 4D outputs on FVD. The superior FVD results are attributed to the use of 3D Gaussian Splatting, which provides a compact and expressive spatial representation that enables more stable and coherent temporal modeling.
<table><tr><td>Method</td><td>FVD-F â</td><td>FVD-Diag â</td><td>FV4D â</td></tr><tr><td>Consistent4D [13]</td><td>1133.93</td><td>741.52</td><td>871.95</td></tr><tr><td>SV3D [32]</td><td>989.53</td><td>526.78</td><td>690.49</td></tr><tr><td>STAG4D [36]</td><td>861.88</td><td>636.83</td><td>546.56</td></tr><tr><td>SV4D [35]</td><td>677.68</td><td>525.65</td><td>614.35</td></tr><tr><td>Dreamgaussian4D [25]</td><td>697.8</td><td>615.68</td><td>638.15</td></tr><tr><td>MVG4D(ours)</td><td>241.99</td><td>201.71</td><td>134.58</td></tr></table>

As shown in Table 3, when compared with state-of-the-art 4D scene generation methods, the 4D rendering results produced by MVG4D achieved significantly lower FVD scores across FVD-F, FVD-Diag, and FV4D, indicating better temporal and view consistency in the generated dynamic content. This improvement is largely attributed to our use of 4D Gaussian point cloud.This is largely due to the use of 3D Gaussian Splatting, which provides a compact yet expressive initialization that better captures spatial structure and supports efficient temporal deformation.

## 4.3.2 Time Efficiency Analysis

Time is a crucial metric for evaluating the practicality of 4D content generation methods, particularly in real-time applications like AR/VR. We benchmark the total time from a single input image to the successful generation of 4D content. As shown in Table 4, our method significantly outperforms prior works in generation speed, benefiting from the use of 4D Gaussian Splatting.

<table><tr><td colspan="2">Table 4: Time efficiency. Benchmarking based on the Objaverse normalized dataset evaluated the processing time required to generate 4D scenes from input images. Compared with the existing</td></tr><tr><td>methods, the proposed method significantly reduces the processing time, which verifies its efficiency. Method</td><td>Time â</td></tr><tr><td>V4D [10]</td><td>6.9 h</td></tr><tr><td>TiNeuVox-B [8]</td><td>28 mins</td></tr><tr><td>DreamGaussian4D [25]</td><td>13 mins</td></tr><tr><td>Stag4D [36] MVG4D (ours)</td><td>9 mins 8 m 46 s</td></tr></table>

Compared to NeRF-based approaches, our lightweight deformation network enables efficient timestamp generation from a static 3D Gaussian point cloud, reducing computational overhead.

## 4.4 Qualitative Results

To evaluate the effectiveness of our method in generating high-quality dynamic 4D content, we conducted qualitative evaluations based on five examples provided by datasets such as Objaverse. As depicted in Figure 3, we provide a visual comparison between a rendered view and a reference image based on a 4D scene representation obtained by 4D GS. From a qualitative perspective, the generated dynamic 4D scenes maintain a consistent appearance and geometric structure across different viewpoints, demonstrating the exceptional capability of MVG4D to handle spatial transformations. Additionally, the generated scenes excel in terms of texture detail and motion realism, thereby showcasing the capability of our proposed method to produce high-quality dynamic content.

<!-- image-->  
Input Image  
Generated 4D Model

Figure 3: Visual representation of the generated results. In the figure, the rendered images at different timestamps and viewpoints are visually compared with the reference image.

Accurate detail rendering and boundary handling are crucial for achieving high-quality dynamic

4D content generation. As shown in Figure 4, we compare the performance of the benchmark Stable Video Diffusion (SVD) [1] method and our proposed MVG4D method. The SVD method exhibits noticeable blurring in high-frequency details, with significant noise and blurred transitions along edges, reducing clarity. In contrast, MVG4D achieves sharper boundaries by preserving structural details and effectively suppressing noise. For boundary handling, the SVD method produces a sawtooth effect along edges, resulting in uneven contours and abrupt transitions, whereas MVG4D delivers smoother and more natural results, enhancing overall visual fidelity.

<!-- image-->  
Figure 4: Detail enlarge comparison diagram. This figure illustrates the comparative performance of the baseline method (SVD) and our proposed method (MVG4D) in terms of detail recovery and boundary treatment.

These results show that MVG4D significantly outperforms current state-of-the-art methods in detail preservation and boundary handling, leading to more realistic and visually compelling 4D dynamic content. This performance gain can be attributed to the use of 3D Gaussian point clouds, which offer richer spatial information than the interval-based sampling strategy in NeRF, thereby enabling higher-quality reconstruction.

## 4.5 Ablation Study

## 4.5.1 Ablation Experiments on Viewpoint Adjustment

We refer to the enhanced diffusion model used in this study as MVIG (Multi-View Image Generation module). Figure 5 shows the results of an ablation study conducted to evaluate the effectiveness of the viewpoint adjustment in improving the fluency and clarity of generated 4D content and enhancing the original diffusion model. In this study, we froze the 3D content generation and 4D content optimization phases to isolate the impact of the subsequent 4D expression. We compared the original diffusion model with three models fine-tuned separately for vertical and horizontal views, all applied to generate dynamic 4D content from the same input image. Our fine-tuned model significantly outperforms the original diffusion model, demonstrating superior action continuity and fluency, with video frames exhibiting less flickering compared to the original model. This improvement, which maintains the temporal consistency of dynamic 4D content, can be attributed to the multi-view image generation module, which controls fluctuations in view parameters and provides more consistent view information for optimizing the 4D GS model.

Table 5: Ablation experiment of MVIG. The clarity and PSNR of the generated content are lower when the multi-view image generation (MVIG) module is removed compared to when it is enabled, which demonstrates the effectiveness and necessity of the proposed method.
<table><tr><td>CLIP-I â</td><td>PSNR â</td></tr><tr><td>w/o MVIG 0.859</td><td>30.54</td></tr><tr><td>Half-MVIG 0.918</td><td>32.87</td></tr><tr><td>Full-MVIG 0.982</td><td>36.44</td></tr></table>

## 4.5.2 Ablation Experiments on MVIG

Table 5 demonstrates the significance of our Multi-View Image Generation module(MVIG) in improving the quality of 4D content. We denote the method without MVIG, which directly uses the input single image for supervised 4D Gaussian Splatting (4D GS) optimization, as âw/o MVIGâ. The method using only the first stage of the MVIG process is labeled âHalf-MVIGâ. Experimental results indicate that the full MVIG, referred to as âFull-MVIGâ, significantly enhances all evaluation metrics. Specifically, âFull-MVIGâ achieves the highest CLIP-I and PSNR scores, indicating that it produces the most semantically consistent and high-definition 4D content.

<!-- image-->  
Figure 5: Impact of multi-view image generation module on model performance. Under identical input images and frame rates, the best results were achieved by the version fine-tuned along the vertical axis, which exhibited smoother transitions and reduced motion artifacts. In contrast, the original diffusion model suffered from flickering effects and abrupt motion changes, highlighting the critical role of the multi-view generation module in achieving high-quality, temporally coherent 4D content.

## 5 Conclusion

We propose MVG4D, a novel framework for efficiently generating dynamic 4D content from a single static image. Central to our method is an image matrix module that synthesizes a temporally coherent and spatially diverse set of multi-view images, providing dense supervision for subsequent 4D representation learning. By integrating this module with dynamic content optimization, our approach significantly improves the temporal continuity, geometric consistency, and visual clarity of the rendered 4D scenes. Experimental results demonstrate that MVG4D excels at reconstructing dynamic content with high fidelity and realism, closely aligning with the appearance of the input image. These advancements lay a strong technical foundation for dynamic 4D content applications in areas such as augmented reality and virtual reality. Future work may explore more complex scenarios, including background control and scene composition, to further enhance the versatility and expressiveness of our framework.

## References

[1] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. CoRR, 2023.

[2] Ang Cao and Justin Johnson. Hexplane: A fast representation for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 130â141, 2023.

[3] Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. Stablevideo: Text-driven consistencyaware diffusion video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23040â23050, 2023.

[4] Zhaoxi Chen, Jiaxiang Tang, Yuhao Dong, Ziang Cao, Fangzhou Hong, Yushi Lan, Tengfei Wang, Haozhe Xie, Tong Wu, Shunsuke Saito, Liang Pan, Dahua Lin, and Ziwei Liu. 3dtopiaxl: Scaling high-quality 3d asset generation via primitive diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26576â 26586, June 2025.

[5] Gene Chou, Yuval Bahat, and Felix Heide. Diffusion-sdf: Conditional generative modeling of signed distance functions. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 2262â2272, October 2023.

[6] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13142â13153, June 2023.

[7] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 7346â 7356, October 2023.

[8] Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Matthias NieÃner, and Qi Tian. Fast dynamic radiance fields with time-aware neural voxels. In SIG-GRAPH Asia 2022 Conference Papers, pages 1â9, 2022.

[9] Sara Fridovich-Keil, Giacomo Meanti, Frederik RahbÃ¦k Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes: Explicit radiance fields in space, time, and appearance. In CVPR, 2023.

[10] Wanshui Gan, Hongbin Xu, Yi Huang, Shifeng Chen, and Naoto Yokoya. V4d: Voxel for 4d novel view synthesis. IEEE Transactions on Visualization and Computer Graphics, 2023.

[11] Jun Gao, Tianchang Shen, Zian Wang, Wenzheng Chen, Kangxue Yin, Daiqing Li, Or Litany, Zan Gojcic, and Sanja Fidler. Get3d: A generative model of high quality 3d textured shapes learned from images. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 31841â31854. Curran Associates, Inc., 2022.

[12] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400, 2023.

[13] Yanqin Jiang, Li Zhang, Jin Gao, Weiming Hu, and Yao Yao. Consistent4d: Consistent 360Â° dynamic object generation from monocular video. In The Twelfth International Conference on Learning Representations, 2024.

[14] Bernhard Kerbl, Georgios Kopanas, Thomas LeimkÂ¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023.

[15] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, et al. Neural 3d video synthesis from multi-view video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5521â5531, 2022.

[16] Huan Ling, Seung Wook Kim, Antonio Torralba, Sanja Fidler, and Karsten Kreis. Align your gaussians: Text-to-4d with dynamic 3d gaussians and composed diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8576â8588, 2024.

[17] Minghua Liu, Ruoxi Shi, Linghao Chen, Zhuoyang Zhang, Chao Xu, Xinyue Wei, Hansheng Chen, Chong Zeng, Jiayuan Gu, and Hao Su. One-2-3-45++: Fast single image to 3d objects with consistent multi-view generation and 3d diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10072â10083, 2024.

[18] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9298â9309, 2023.

[19] Yu-Lun Liu, Chen Gao, Andreas Meuleman, Hung-Yu Tseng, Ayush Saraf, Changil Kim, Yung-Yu Chuang, Johannes Kopf, and Jia-Bin Huang. Robust dynamic radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

[20] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Generating multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453, 2023.

[21] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9970â9980, 2024.

[22] Luke Melas-Kyriazi, Iro Laina, Christian Rupprecht, and Andrea Vedaldi. Realfusion: 360deg reconstruction of any object from a single image. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8446â8455, 2023.

[23] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021.

[24] Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. Deepsdf: Learning continuous signed distance functions for shape representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.

[25] Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao, Gang Zeng, and Ziwei Liu. Dreamgaussian4d: Generative 4d gaussian splatting. arXiv preprint arXiv:2312.17142, 2023.

[26] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and BjÂ¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684â10695, 2022.

[27] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110, 2023.

[28] Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, et al. Text-to-4d dynamic scene generation. In Proceedings of the 40th International Conference on Machine Learning, pages 31915â31929, 2023.

[29] Qi Sun, Zhiyang Guo, Ziyu Wan, Jing Nathan Yan, Shengming Yin, Wengang Zhou, Jing Liao, and Houqiang Li. Eg4d: Explicit generation of 4d object without score distillation. arXiv preprint arXiv:2405.18132, 2024.

[30] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018.

[31] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[32] Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. In European Conference on Computer Vision, pages 439â457. Springer, 2025.

[33] Feng Wang, Zilong Chen, Guokang Wang, Yafei Song, and Huaping Liu. Masked space-time hash encoding for efficient dynamic scene reconstruction. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 70497â70510. Curran Associates, Inc., 2023.

[34] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20310â20320, 2024.

[35] Yiming Xie, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, and Varun Jampani. Sv4d: Dynamic 3d content generation with multi-frame and multi-view consistency. arXiv preprint arXiv:2407.17470, 2024.

[36] Yifei Zeng, Yanqin Jiang, Siyu Zhu, Yuanxun Lu, Youtian Lin, Hao Zhu, Weiming Hu, Xun Cao, and Yao Yao. Stag4d: Spatial-temporal anchored generative 4d gaussians. In European Conference on Computer Vision, pages 163â179. Springer, 2025.

[37] Haiyu Zhang, Xinyuan Chen, Yaohui Wang, Xihui Liu, Yunhong Wang, and Yu Qiao. 4diffusion: Multi-view video diffusion model for 4d generation. Advances in Neural Information Processing Systems, 37:15272â15295, 2024.

[38] Yuyang Zhao, Zhiwen Yan, Enze Xie, Lanqing Hong, Zhenguo Li, and Gim Hee Lee. Animate124: Animating one image to 4d dynamic scene. arXiv preprint arXiv:2311.14603, 2023.