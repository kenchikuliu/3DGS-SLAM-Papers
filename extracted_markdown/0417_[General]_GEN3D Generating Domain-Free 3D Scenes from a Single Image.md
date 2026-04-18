# GEN3D:GENERATING DOMAIN-FREE 3D SCENES FROM A SINGLE IMAGE

Yuxin Zhang1,2, Ziyu Lu1,2, Hongbo Duan1,2, Keyu Fan1 Pengting Luo2, Peiyu Zhuang2, Mengyu Yang1,2, Houde Liu1,\*

1 Shenzhen International Graduate School, Tsinghua University, Shenzhen, China 2 Central Media Technology Institute, Huawei, Shenzhen, China

## ABSTRACT

Despite recent advancements in neural 3D reconstruction, the dependence on dense multi-view captures restricts their broader applicability.Additionally, 3D scene generation is vital for advancing embodied AI and world models, which depend on diverse, highquality scenes for learning and evaluation.In this work, we propose Gen3d, a novel method for generation of high-quality, wide-scope, and generic 3D scenes from a single image. After the initial point cloud is created by lifting the RGBD image, Gen3d maintains and expands its world model.The 3D scene is finalized through optimizing a Gaussian splatting representation. Extensive experiments on diverse datasets demonstrate the strong generalization capability and superior performance of our method in generating a world model and Synthesizing high-fidelity and consistent novel views.

Index Termsâ Novel View Synthesis, 3D Scene Generation, world model

## 1. INTRODUCTION

With the advent of commercial mixed reality platforms and the rapid innovations in 3D graphics technology, high-quality 3D scene generation has become one of the most important problem in computer vision for applications such as immersive media, robotics, autonomous driving, and embodied AI.Previous approaches mainly include Geometric reconstruction and video generation methods.Geometric reconstruction methods suffer from severe artifacts at viewpoints unseen in input views, while video generation methods lack good geometric consistency in their outputs.

In this work, we propose a pipeline called Gen3d.To address the artifact generation issue of geometric reconstruction methods, Gen3d adopts a hierarchical strategy and leverages Stable Diffusion [1] and 3D Gaussian splatting [2] to generate a wide variety of highquality 3D scenes with geometric consistency from multiple types of inputs, such as text, RGB images, and RGBD images.

Gen3d first decomposes the input image into foreground and background layers via depth maps (input or from depth models).The background is inpainted via Lama-ControlNet inpainting[3] guided image inpainting, and initial point clouds are generated for both layers. The camera then moves along a predefined trajectory. At each pose, visible parts of the point cloud are projected onto the new view, and a Stable Diffusion-based inpainting network synthesizes a complete image from this partial projection. The result, along with its depth map, is lifted back to 3D space, with point positions refined via an alignment algorithm. Finally, the aggregated point cloud initializes a 3D Gaussian splatting representation, which resolves depth inconsistencies and enables high-quality novel view synthesis beyond traditional point-based methods.

In summary, our contributions are as follows.

â¢ We introduce Gen3d, a domain-free,high-quality 3D scene generation method, achieving better domain generalization in 3D scene generation by leveraging the power of Stable Diffusion, depth estimation, and explicit 3D representation.

â¢ To generate multi-view images, we adopt a hierarchical strategy that first constructs a point cloud as a geometric prior for each novel view to generate novel view images. The generated images are then seamlessly integrated to form a coherent 3D scene.

â¢ Our model provides users with the ability to create 3D scenes in various ways by supporting different input types, such as text, RGB, and RGBD.

## 2. RALETED WORK

Novel view generation.Given a single image, early methods infer 3D scene representations and use them to render novel views. These representations include point clouds [4, 5], multi-plane images [6, 7], depth maps[8], and meshes [9]. Despite enabling fast rendering, these representations limit camera movement due to their finite spatial extent. In addition, they only supported generating views within small viewpoint changes w.r.t. the input image, as they only built single static scene representations that do not go beyond the input image. Gen3d focus on a generative task to support creating many connected scenes rather than a single one.

3D scene generation.Recently, advancements in scene generation have predominantly centered on modeling isolated, local 3D environments. A significant number of these efforts explicitly target indoor settings [10, 11], while another line of research concentrates predominantly on outdoor scenarios [12]. This thematic specialization limits the generalizability of such approaches across diverse environmental contexts.

Video generation.Recent improvements in video generation [13] have led to interest in whether these models can also be used as scene generators. Several works have attempted to add camera control, allowing a user to âmoveâ through the scene [14]. While these are promising, they currently do not guarantee 3D consistency.

Fast 3D scene representations.3D Gaussian Splatting(3DGS) excels in high-fidelity real-time rendering by explicitly modeling scenes as anisotropic 3D Gaussiansâeach defined by position, covariance, opacity, and spherical harmonics coefficients. Unlike implicit volumetric methods requiring costly ray marching, 3DGS enables efficient GPU-based rasterization, achieving real-time highresolution view synthesis. Its adaptive optimization dynamically adjusts Gaussians to capture fine details. Leveraging these advantages (efficiency, explicit control, and rendering quality), we adopt

3DGS as our core representation for sparse-input 3D reconstruction.

## 3. METHODOLOGY

Given an input image or text prompt, our purpose is to generate realistic and high-quality 3D scenes that are conditioned on this input. In scenarios where only a text prompt is provided, Gen3d is capable of producing scenes that are semantically related to the given text. Furthermore, Gen3d excels in creating specified scenes based on text prompts while preserving the stylistic elements of the input image.

## 3.1. Single-view Layer Generation

To decompose a scene into layered representations from a single view, we propose a hybrid approach combining depth-prior-guided segmentation with diffusion-based inpainting.As illustrated in Fig. 1, our framework first generates foreground object masks $\mathcal { M } _ { f g }$ and background masks $\mathcal { M } _ { b g }$ using depth-aware Segment Anything Model (SAM)[15], then reconstructs occluded regions in the background layer through text-conditioned inpainting.

Depth-guided Mask Generation.Given an input image $I \in \mathbf { \Sigma }$ $\mathbb { R } ^ { \tilde { H } \times W \times 3 }$ , we estimate its depth map $D \in \mathbb { R } ^ { H \times W }$ using Moge2[16]. The SAM model produces $N$ candidate masks $\{ \mathbf { M } _ { i } \} _ { i = } ^ { N }$ 1 with associated confidence scores $\{ s _ { i } \} _ { i = 1 } ^ { N }$ . We filter these masks through:

$$
\mathcal { V } = \{ \mathbf { M } _ { i } | s _ { i } > \tau _ { i o u } \land \mathrm { a r e a } ( \mathbf { M } _ { i } ) \in [ A _ { m i n } , A _ { m a x } ] \}\tag{1}
$$

where $\tau _ { i o u } = 0 . 8 5$ and $A _ { m i n / m a x }$ are set as 0.5% / 60% of image area. Each valid mask is then classified by depth consistency:

$$
\mathbf { M } _ { f g } = \bigcup _ { \mathbf { M } _ { i } \in \mathcal { V } } \{ \mathbf { M } _ { i } \ | \ \operatorname* { m e d i a n } ( D | _ { \mathbf { M } _ { i } } ) < \theta _ { d } \}\tag{2}
$$

with $\theta _ { d }$ dynamically set as the 35% of D. The background mask is derived as $\mathbf { M } _ { b g } = \mathbf { 1 } - \mathbf { M } _ { f g }$

Occlusion-aware Inpainting. For regions $\Omega = \mathbf { M } _ { f g } \cap \{ D > \theta _ { d } \}$ we employ Lama-ControlNet with two key modifications:

(1)Depth-conditioned attention:The cross-attention layers in Stable Diffusion are modulated by normalized depth values $\tilde { D } =$ $( D - D _ { m i n } ) / ( D _ { m a x } - D _ { m i n } )$

(2)Textual prompting: Background inpainting uses the template âhigh-resolution [scene category] background with [dominant colors] colors, photorealistic, $8 \bar { \mathsf { K } } ^ { \prime \prime }$ where bracketed terms are auto-filled by CLIP[17].

## 3.2. world generation

To generate a multi-view consistent 3D point cloud, we first create the initial point cloud.Subsequently, while moving the camera,we aggregate both the new points and initial points by moving back and forth between the 3D space and the camera plane, with the reconstruction results situated in a unified world coordinate system.The overall process of point cloud construction is illustrated in Figure 1. Initialization. A point cloud generation starts from lifting the pixels of the initial image. If the user provides a text prompt as input, a latent diffusion model is utilized to generate an image relevant to the given text, and the metric depth map is estimated using Moge2. We denote the generated or received RGB image and the corresponding depth map as $\mathbf { I } _ { 0 } \in \mathbb { R } ^ { 3 \times H \times W }$ and $\mathbf { D } _ { 0 } \in \mathbb { R } ^ { H \times W }$ , respectively, where H and W represent the height and the width of the image. The camera intrinsic matrix and the extrinsic matrix of ${ \bf { I } } _ { 0 }$ are denoted as K and $\mathbf { P } _ { 0 } ,$ , respectively. For the case where $\mathbf { I } _ { 0 }$ and $\mathbf { D } _ { 0 }$ are generated from the diffusion model, we set the values of K and $\mathbf { P } _ { 0 }$ by convention according to the size of the image.

From the RGBD image $[ \mathbf { I } _ { 0 } , \mathbf { D } _ { 0 } ]$ , we lift the pixels into the 3D space, where the lifted pixels form a point cloud. The generated initial point cloud using the first image is defined as $\mathcal { P } _ { 0 }$

$$
\mathcal { P } _ { 0 } = f _ { 2 \mathrm { d }  3 \mathrm { d } } ( \lbrack \mathbf { I } _ { 0 } , \mathbf { D } _ { 0 } \rbrack , \mathbf { K } , \mathbf { P } _ { 0 } ) ,\tag{3}
$$

where $f _ { 2 \mathrm { d } \to 3 \mathrm { d } }$ is the function that lifts pixels from the RGBD image [I, D] to the point cloud.

Point Cloud Augmentation and aggregation. We sequentially attach points to the original point cloud to construct a large-scale 3D scene. Specifically, we define a counterclockwise rotating camera trajectory trajectory of length N , where $\mathbf { P } _ { i }$ denotes the camera position and pose at the i-th index. At each step, we inpaint and lift the missing pixels.We leverage the representational power of Stable Diffusion for the image inpainting task. Specifically,At step i, we first move and rotate the camera from the previous position $\mathbf { P } _ { i - 1 }$ to $\mathbf { P } _ { i } .$ . The coordinate system is transformed from world coordinates to the current camera coordinates, followed by projection onto the camera plane using the intrinsic matrix K and the extrinsic matrix $\mathbf { P } _ { i }$ . We denote the projected image at camera $\mathbf { P } _ { i }$ as ${ \hat { \mathbf { I } } } _ { i } .$ Due to the change in camera position and pose, certain regions in $\hat { \mathbf { I } } _ { i }$ cannot be filled from the existing point cloud. We define a binary mask Mi to indicate the filled regions: Mi equals 1 if the corresponding pixel is filled by existing points, and 0 otherwise. The Stable Diffusion inpainting model (S) is employed to generate a realistic image Ii from the incomplete image $\hat { \mathbf { I } } _ { i }$ and the mask Mi. The corresponding depth map DË i is estimated using Moge2.

Note that the monocular depth estimation model provides metric depth values. We then lift pixels to 3D space using the inpainted image Ii and its corresponding depth map $\mathbf { D } _ { i }$ . To save memory consumption and represent efficiently, only pixels in the inpainted regions $( \mathbf { M } _ { i } = 0 )$ are lifted.

Compared to approaches that train generative models to simultaneously produce both RGB and depth maps,such as RGBD2[18] employing off-the-shelf depth estimation methods yields more accurate and generalizable depth maps, as these models are trained on large and diverse datasets. However, since $\mathbf { D } _ { 0 } , \mathbf { D } _ { 1 } , \dots , \mathbf { D } _ { i - 1 }$ are not considered when estimating Di, an inconsistency arises when integrating new points $\hat { \mathcal { P } } _ { i } ^ { \phantom { \dagger } }$

To address this issue, we adjust the points in $\hat { \mathcal { P } } _ { i }$ within the 3D space to ensure smooth integration between the existing point cloud $\mathcal { P } _ { i - 1 }$ and the new points $\hat { \mathcal { P } } _ { i }$ . Specifically, we extract the region where the mask boundary changes $( | \nabla { \mathbf M } _ { i } | > 0 )$ to identify corresponding points in both $\mathcal { P } _ { i - 1 }$ and $\hat { \mathcal { P } } _ { i }$ . We then compute a displacement vector from $\hat { \mathcal { P } } _ { i }$ to $\mathcal { P } _ { i - 1 }$ . However, naively moving points may distort the geometry of the lifted point cloud and cause misalignment with the inpainted image. To mitigate this, we impose constraints on point movement and employ an interpolation algorithm to preserve the overall structure.

First, we constrain each point in $\hat { \mathcal { P } } _ { i }$ to move along the ray originating from the camera center to its corresponding pixel. We locate the closest point in $\mathcal { P } _ { i - 1 }$ along this ray and quantify the required depth adjustment. This constraint ensures that the visual content of the RGB image Ii remains consistent despite 3D point adjustments. Second, we assume depth values remain unchanged on the opposite side of the mask boundary. For points without ground-truth correspondences (i.e., where ${ { \mathbf { M } } _ { i } } \ = \ 0 )$ , we compute depth changes via linear interpolation. Smooth interpolation alleviates artifacts caused by abrupt movements.

The aligned point cloud is combined with the original as follows:

$$
\mathcal { P } _ { i } = \mathcal { P } _ { i - 1 } \cup \mathcal { W } \left( \mathcal { \hat { P } } _ { i } \right) ,\tag{4}
$$

<!-- image-->  
Fig. 1. Gen3d pipeline.Initially, the input 2D image is segmented into two distinct components: the foreground objects and the background. We adopt methodologies such as the Stable Diffusion model and monocular depth estimation to enhance point cloud coverage and facilitate the construction of larger-scale scenes.Subsequently, we employ the point cloud alongside the reprojected images to optimize a set of Gaussian splats, further refining the resulting 3D scene.

where W denotes the movement and interpolation operation. This process is repeated N times to construct the final point cloud $\mathcal { P } _ { N } .$ Through reprojection, $\mathcal { P } _ { N }$ delivers high-quality, multi-view consistent images. The overall procedure for constructing $\mathcal { P } _ { N }$ from $[ \mathbf { I } _ { 0 } , \mathbf { D } _ { 0 } ] ,$ , K.

## 3.3. Rendering with gaussian splatting

After the point cloud is constructed, we train a 3D Gaussian Splatting model using the point cloud and the projected images. The centers of the Gaussian splats are initialized from the input point cloud, while the volume and position of each point are optimized under the supervision of the ground truth projected images.The loss function is constructed as a weighted combination of L1 loss and SSIM loss.

Initializing with $\mathcal { P } _ { N }$ accelerates network convergence and encourages the model to focus on reconstructing fine-grained details. For training, we use an additional set of M images alongside the original $( N + 1 )$ images employed in point cloud generation, as the initial set alone is insufficient for producing plausible results. These M new images and their corresponding masks are generated by reprojecting $\mathcal { P } _ { N }$ under a new camera sequence $\mathbf { P } _ { N + 1 } , \hdots , \mathbf { P } _ { N + M } \mathrm { : }$

$$
{ \bf I } _ { i } , { \bf M } _ { i } = f _ { 3 d  2 d } ( \mathcal { P } _ { N } , { \bf K } , { \bf P } _ { i } ) , \quad i = N + 1 , \ldots , N + M .\tag{5}
$$

Note that we do not perform inpainting on Ii during the optimization of the Gaussian splats. Instead, when computing the loss function, we only consider the valid image regions where the mask value is 1. This prevents the model from learning artifacts in the reprojected images. Since each point is represented as a Gaussian distribution, missing pixels are naturally filled during rendering, and the resulting rasterized image becomes plausible after training.

## 4. EXPERIMENTS

## 4.1. Implementation details.

The modules we used to construct Gen3d can be either trained using manual design or brought from off-the-shelf models.We use pretrained large-scale off-the-shelf models to compose the whole network to maximize the generalization capability of the network. Specifically, we use the same text prompt input for the Stable Diffusion if the first image is generated from the text. If the input format is a RGB(D) image without text, we use LAVIS[24] to generate the caption according to the image and place it in the diffusion inpainting model to generate consistent content. For the camera trajectory that we use to construct the point cloud $( \{ \mathbf { P } _ { i } \} _ { i = 0 } ^ { N } )$ , we create several types of camera trajectory presets in advance, and different types of trajectories were used for different tasks.

## 4.2. World Generation

We test Gen3d on WorldScore [25] static benchmark on world generation. WorldScore consists of 2,000 static test examples that span diverse worlds. The metrics evaluate the controllability and quality of video generation. Specifically, we use âCamera Controlâ, âObject Controlâ, and âContent Alignmentâ to judge how the model adhere to viewpoint instructions and text prompts. We use â3D Consistencyâ, âPhotometric Consistencyâ, âStyle Consistencyâ, and âSubjective Qualityâ to evaluate the consistency and quality of generated content. Finally, an average score is presented to show the overall performance. We compare six 3D generation methods in the existing benchmark. The scores are reported in Table1.Gen3d achieves the highest score on this benchmark.The score shows that Gen3d has competitive performance on camera control and 3D consistency, compared with 3D Models. Our Photometric consistency score is the highest among all methods, further demonstrating the visual quality of our generated videos.

Table 1. WorldScore Benchmark Comparison. Abbreviations: Ctrl=Controllability,Align=Alignment,Consist=Consistency,Photo=Photometric.The top three rankings from highest to lowest are marked with
<table><tr><td rowspan="2">Models</td><td rowspan="2">Worldscore</td><td colspan="3">Controllability</td><td colspan="4">Quality</td></tr><tr><td>Camera Ctrl</td><td>Object Ctrl</td><td>Content Align</td><td>3D Consist</td><td>Photo Consist</td><td>Style Consist</td><td>Subjective Qual</td></tr><tr><td>SceneScape[19]</td><td>50.73</td><td>84.99</td><td>47.44</td><td>28.64</td><td>76.54</td><td>62.88</td><td>21.85</td><td>32.75</td></tr><tr><td>Text2Room[11]</td><td>62.10</td><td>94.01</td><td>38.93</td><td>50.79</td><td>88.71</td><td>88.36</td><td>37.23</td><td>36.69</td></tr><tr><td>LucidDreamer[20]</td><td>70.40</td><td>88.93</td><td>41.18</td><td>75.00</td><td>90.37</td><td>90.20</td><td>48.10</td><td>58.99</td></tr><tr><td>WonderJourney[21]</td><td>63.75</td><td>84.60</td><td>37.10</td><td>35.54</td><td>80.60</td><td>79.03</td><td>62.82</td><td>666.56</td></tr><tr><td>InvisibleStitch[22]</td><td>61.12</td><td>93.20</td><td>36.51</td><td>29.53</td><td>88.51</td><td>89.19</td><td>32.37</td><td>58.50</td></tr><tr><td>WonderWorld[23]</td><td>72.69</td><td>92.98</td><td>51.76</td><td>71.25</td><td>86.87</td><td>85.56</td><td>70.57</td><td>49.81</td></tr><tr><td>Gen3d(ours)</td><td>75.05</td><td>99.77</td><td>41.00</td><td>66.73</td><td>91.79</td><td>91.11</td><td>77.01</td><td>57.92</td></tr></table>

<!-- image-->  
Fig. 2. Comparisons of multi-view generation results across different methods.The images are sourced from the COCO dataset, the WorldScore Benchmark, and web-sourced images.

To comprehensively evaluate our Gen3d model for multi-view 3D generation, we conducted qualitative comparisons with topperforming baselines in our benchmark: WonderWorld and LucidDreamer. These models were selected for their state-of-the-art performance.WonderWorld is optimized only for open-landscape outdoor scenes and fails at indoor generation. Thus, indoor comparisons only include LucidDreamer and Gen3d.

As shown in Fig. 2, LucidDreamer has two critical flaws under large camera movements: noticeable geometric holes in complex occlusions, and inconsistent generation quality across views. In contrast, Gen3d via our layered strategy decomposes scenes into layers and optimizes their consistency independently, effectively synthesizing occluded regions and maintaining texture and shape uniformity. For outdoor scenes, WonderWorld, despite its strengths in clear-sky outdoor environments, still produces obvious floating artifacts during camera motion owing to the use of semantic hard-coding for reconstruction.Gen3d, however, maintains stable, artifact-free generation and accurate geometry in these challenging outdoor scenarios.

## 5. CONCLUSION

This study addresses the limitation of traditional neural 3D reconstructionâreliance on dense multi-view capturesâby proposing Gen3d, a novel pipeline for high-quality, wide-scope 3D scene generation from a single input.

Gen3d includes: Stable Diffusion for photorealistic 2D synthesis, 3DGS for efficient 3D representation, and point cloud geometric guidance for cross-view consistency.It proceeds in steps: segmentation with initial point cloud generation, iterative point cloud expansion, diffusion-based novel view synthesis,3DGS optimization to eliminate holes and enable real-time rendering.

Future work will focus on accelerating point cloud expansion and extending Gen3d to dynamic 3D scene generation.

## 6. REFERENCES

[1] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and Â¨ George Drettakis, â3d gaussian splatting for real-time radiance field rendering.,â ACM Trans. Graph., vol. 42, no. 4, pp. 139â 1, 2023.

[2] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer, âHigh-resolution image synthesis Â¨ with latent diffusion models,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10684â10695.

[3] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky, âResolution-robust large mask inpainting with fourier convolutions,â arXiv preprint arXiv:2109.07161, 2021.

[4] Chris Rockwell, David F Fouhey, and Justin Johnson, âPixelsynth: Generating a 3d-consistent experience from a single image,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 14104â14113.

[5] Jing Yu Koh, Harsh Agrawal, Dhruv Batra, Richard Tucker, Austin Waters, Honglak Lee, Yinfei Yang, Jason Baldridge, and Peter Anderson, âSimple and effective synthesis of indoor 3d scenes,â in Proceedings of the AAAI conference on artificial intelligence, 2023, vol. 37, pp. 1169â1178.

[6] Richard Tucker and Noah Snavely, âSingle-view view synthesis with multiplane images,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 551â560.

[7] Tewodros Amberbir Habtegebrial, Varun Jampani, Orazio Gallo, and Didier Stricker, âGenerative view synthesis: From single-view semantics to novel-view images,â Advances in neural information processing systems, vol. 33, pp. 4745â 4755, 2020.

[8] Meng-Li Shih, Shih-Yang Su, Johannes Kopf, and Jia-Bin Huang, â3d photography using context-aware layered depth inpainting,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 8028â8038.

[9] Ronghang Hu, Nikhila Ravi, Alexander C Berg, and Deepak Pathak, âWorldsheet: Wrapping the world in a 3d sheet for view synthesis from a single image,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 12528â12537.

[10] Nantheera Anantrasirichai and David Bull, âArtificial intelligence in the creative industries: a review,â Artificial intelligence review, vol. 55, no. 1, pp. 589â656, 2022.

[11] Lukas Hollein, Ang Cao, Andrew Owens, Justin Johnson, Â¨ and Matthias NieÃner, âText2room: Extracting textured 3d meshes from 2d text-to-image models,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 7909â7920.

[12] Matt Deitke, Eli VanderBilt, Alvaro Herrasti, Luca Weihs, Kiana Ehsani, Jordi Salvador, Winson Han, Eric Kolve, Aniruddha Kembhavi, and Roozbeh Mottaghi, âProcthor: Large-scale embodied ai using procedural generation,â Advances in Neural Information Processing Systems, vol. 35, pp. 5982â5994, 2022.

[13] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[14] Zheng Zhu, Xiaofeng Wang, Wangbo Zhao, Chen Min, Nianchen Deng, Min Dou, Yuqi Wang, Botian Shi, Kai Wang, Chi Zhang, et al., âIs sora a world simulator? a comprehensive survey on general world models and beyond,â arXiv preprint arXiv:2405.03520, 2024.

[15] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Â¨ Chloe Rolland, Laura Gustafson, et al., âSam 2: Segment anything in images and videos,â arXiv preprint arXiv:2408.00714, 2024.

[16] Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng Xiang, Zelong Lv, Guangzhong Sun, Xin Tong, and Jiaolong Yang, âMoge-2: Accurate monocular geometry with metric scale and sharp details,â arXiv preprint arXiv:2507.02546, 2025.

[17] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al., âLearning transferable visual models from natural language supervision,â in International conference on machine learning. PmLR, 2021, pp. 8748â8763.

[18] Tackgeun You, Mijeong Kim, Jungtaek Kim, and Bohyung Han, âGenerative neural fields by mixtures of neural implicit functions,â Advances in Neural Information Processing Systems, vol. 36, pp. 20352â20370, 2023.

[19] Rafail Fridman, Amit Abecasis, Yoni Kasten, and Tali Dekel, âScenescape: Text-driven consistent scene generation,â Advances in Neural Information Processing Systems, vol. 36, pp. 39897â39914, 2023.

[20] Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin Lee, and Kyoung Mu Lee, âLuciddreamer: Domain-free generation of 3d gaussian splatting scenes,â arXiv preprint arXiv:2311.13384, 2023.

[21] Hong-Xing Yu, Haoyi Duan, Junhwa Hur, Kyle Sargent, Michael Rubinstein, William T Freeman, Forrester Cole, Deqing Sun, Noah Snavely, Jiajun Wu, et al., âWonderjourney: Going from anywhere to everywhere,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 6658â6667.

[22] Paul Engstler, Andrea Vedaldi, Iro Laina, and Christian Rupprecht, âInvisible stitch: Generating smooth 3d scenes with depth inpainting,â in 2025 International Conference on 3D Vision (3DV). IEEE, 2025, pp. 457â468.

[23] Hong-Xing Yu, Haoyi Duan, Charles Herrmann, William T Freeman, and Jiajun Wu, âWonderworld: Interactive 3d scene generation from a single image,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 5916â5926.

[24] Dongxu Li, Junnan Li, Hung Le, Guangsen Wang, Silvio Savarese, and Steven C. H. Hoi, âLavis: A library for language-vision intelligence,â 2022.

[25] Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and Jiajun Wu, âWorldscore: A unified evaluation benchmark for world generation,â arXiv preprint arXiv:2504.00983, 2025.