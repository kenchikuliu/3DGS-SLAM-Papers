# ZERO-SHOT VISUAL GROUNDING IN 3D GAUSSIANS VIA VIEW RETRIEVAL

Liwei Liao1,2, Xufeng Li3, Xiaoyun Zheng1,2, Boning Liu2, Feng Gao1â, Ronggang Wang2

1 Peking University, 2 Peng Cheng Laboratory, 3 City University of Hongkong

## ABSTRACT

3D Visual Grounding (3DVG) aims to locate objects in 3D scenes based on text prompts, which is essential for applications such as robotics. However, existing 3DVG methods encounter two main challenges: first, they struggle to handle the implicit representation of spatial textures in 3D Gaussian Splatting (3DGS), making per-scene training indispensable; second, they typically require larges amounts of labeled data for effective training. To this end, we propose Grounding via View Retrieval (GVR), a novel zero-shot visual grounding framework for 3DGS to transform 3DVG as a 2D retrieval task that leverages object-level view retrieval to collect grounding clues from multiple views, which not only avoids the costly process of 3D annotation, but also eliminates the need for per-scene training. Extensive experiments demonstrate that our method achieves state-of-the-art visual grounding performance while avoiding per-scene training, providing a solid foundation for zero-shot 3DVG research. Video demos can be found in https://github.com/leviome/ GVR_demos.

Index Termsâ 3D Visual Grounding, View Retrieval, Scene Understanding

## 1. INTRODUCTION

3D Gaussian Splatting (3DGS) [1] offers significant advantages over traditional 3D reconstruction methods in terms of reconstruction speed, quality, and rendering efficiency. However, because 3DGS adopts a semi-implicit representation with explicit geometric structures and implicit spatial textures, it is difficult to directly apply previous point-cloud or radiance-field methods to 3DGS.

In recent years, there has been a growing body of work on developing scene understanding algorithms [2â10] specifically for 3DGS. These approaches can be categorized into two main types: (1) supervised 3DVG methods such as SceneSplat [10] that require annotated 3D data for training and perform end-to-end inference, and (2) per-scene training 3DVG methods such as LangSplat [11] that train a language Gaussian field for each specific scene. However, both the two approaches face different limitations: (1) For SceneSplat, pre-training is conducted only on 7,000 indoor scenes, so its generalization ability is limited to several indoor environments. To expand its generalization requires a massive amount of 3D annotated data, which is currently expensive. (2) For LangSplat, hour-scale per-scene preparation (including preprocessing and training) are required, and the query stage is also computationally complex. These two limitations severely restrict the application of 3DVG in the 3DGS domain, especially the requirement for hours of per-scene training, which greatly undermines its practicality.

To address the two limitations, we propose a retrievalbased 3D Gaussian visual grounding method that reformulates the 3DVG task as a 2D retrieval problem, enabling effective visual grounding in 3D Gaussian scenes. For the first limitation, our work is training-free by leveraging existing mature 2D visual foundation models to accomplish visual perception, thereby largely preserving the generalization ability of these 2D models. For the second limitation, we replace per-scene training with a knowledge book approach, which not only greatly reduces preparation time for each scene but also accelerates the query process. Our main contributions are summarized as follows:

â¢ We introduce GVR, to the best of our knowledge, the first framework capable of zero-shot visual grounding in 3D Gaussian scenes.

â¢ We propose a view retrieval mechanism for 3DVG, which eliminates the need for 3D annotated data and enables high-quality visual grounding using only existing mature 2D visual foundation models.

â¢ Extensive experiments demonstrate that our approach achieves state-of-the-art performance in zero-shot 3D visual grounding of 3DGS, while significantly saving training time.

## 2. METHODOLOGY

Our method aims to address the problem of zero-shot 3D visual grounding specifically for 3DGS. The overall process can be formulated as:

$$
{ \bf I } _ { t a r } = \mathrm { G V R } ( Q ; { \bf G } )\tag{1}
$$

where $\mathbf { I } _ { t a r } \in \mathbb { R } ^ { N \times 1 }$ indicates the flag of all Gaussians that 1 denotes target and 0 denotes background, Q is the text query

$\mathrm { ( e . g . , ~ a ~ }$ red apple) describing the desired object or region, $\mathbf { G } ~ \in ~ \mathbb { R } ^ { N \times 5 9 }$ denotes the 3D Gaussian scene containing N Gaussians, and GVR(Â·) refers to our proposed 3DVG method.

## 2.1. (A) Preparation: Knowledge Books Building

As shown in Fig. 1 (A), we first prepare Semantic Vector Book (SVB) and Depth Book (DB) $\left\{ \mathbf { B } _ { s } , \mathbf { B } _ { d } \right\}$ . Since the 3D scene G is reconstructed from multi-view images and camera parameters {V, C}, all semantic information about the scene can be inferred from these multi-view images. Based on this observation, we construct a SVB $\mathbf { B } _ { s } \in \mathbb { R } ^ { n \times m \times c }$ to store the sceneâs semantic information. Specifically, we use SAM [12] to segment each view and obtain multiple object masks, then encode each object patch into a semantic vector using CLIPâs [13] image encoder, as:

$$
\mathbf { B } _ { s } ^ { ( i ) ( j ) } = \mathcal { E } _ { i m g } ( \mathrm { S A M } ( v _ { i } ) ^ { ( j ) } )\tag{2}
$$

where SAM(Â·) denotes the SAM model that segments the input image into m object patches, $\mathcal { E } _ { i m g } ( \cdot )$ represents CLIPâs image encoder that encodes each object patch into a cdimensional semantic vector (typically $c = 5 1 2 )$ , and $\mathbf { B } _ { s } ^ { ( i ) ( j ) } \in$ $\mathbb { R } ^ { c }$ is the semantic vector for the j-th object in the i-th view. Meanwhile, we obtain a depth map for each view via 3DGS depth rendering, and aggregate these to build a DB $\mathbf { B } _ { d } ~ \in ~ \mathbb { R } ^ { n \times H \times W }$ for storing the sceneâs depth information. This step can be formulated as:

$$
\mathbf { B } _ { d } ^ { ( i ) } = { \mathcal { D } } ( c _ { i } ; \mathbf { G } )\tag{3}
$$

where $\mathcal { D } ( \cdot ; { \bf G } )$ denotes the depth rendering of the 3D Gaussian scene G, ci represents the camera parameters for the i-th view, and $\mathbf { B } _ { d } ^ { ( i ) } \in \mathbb { R } ^ { H \times W }$ is the depth map for the i-th view, which has the same resolution to view $v _ { i }$

## 2.2. (B) Query: Retrieval For Localizing (RFL)

As shown in Fig. 1 (B), this stage is for retrieving relevant patches of views for a given textual query and localizing objects within those clues. First, we encode the text query Q (such as âa red appleâ) into a semantic vector using CLIPâs text encoder. We then compute the similarity between this vector and each item of SVB $\mathbf { B } _ { s } ,$ , selecting the patch with the highest similarity in each view. This allows us to obtain the 2D localization $L _ { 2 D }$ of the target object in each view. This step can be formulated as:

$$
L _ { 2 D } ^ { ( i ) } = \mathrm { p o s i t i o n } ( \arg \operatorname* { m a x } _ { j } \mathcal { S } ( \mathcal { E } _ { t e x t } ( Q ) , { \bf B } _ { s } ^ { ( i ) } ) )\tag{4}
$$

where $\mathcal { E } _ { t e x t }$ denotes text encoder, $\boldsymbol { \mathcal { S } } ( \cdot , \cdot )$ denotes the cosine similarity function, and $L _ { 2 D } ^ { ( i ) } \in \mathbb { R } ^ { 2 }$ represents the 2D coordinates of the target object in the i-th view. By incorporating depth information, we can map the 2D location $L _ { 2 D } ^ { ( i ) }$ to 3D space, thereby obtaining the 3D position $L _ { 3 D } ^ { ( i ) }$ of the target object. This step can be formulated as:

$$
L _ { 3 D } ^ { ( i ) } = \mathcal { P } ^ { - 1 } ( L _ { 2 D } ^ { ( i ) } , \mathbf { B } _ { d } ^ { ( i ) } , c _ { i } )\tag{5}
$$

where $\mathcal { P } ^ { - 1 } ( \cdot , \cdot , \cdot )$ denotes the back-projection function that maps 2D coordinates to 3D space using the depth map and camera parameters, and $L _ { 3 D } ^ { ( i ) } \in \mathbb { R } ^ { 3 }$ represents the 3D coordinates of the target object in the i-th view. Since each view can yield a 3D position and CLIP itself exhibits instability, erroneous retrieval results may occur. Therefore, we design a Multi-view Stereo Voting strategy: we evaluate the Euclidean distances among the 3D positions obtained from each view, and the location indicated by the majority of views is regarded as the final 3D position $L _ { 3 D } \in \mathbb { R } ^ { 3 }$

## 2.3. (C) Online Segmentation

As shown in Fig. 1 (C), this stage is for segmenting the target Gaussians based on the 3D position $L _ { 3 D }$ . First, we render a target-centered birdâs-eye view (BEV) $v _ { B E V }$ based on $L _ { 3 D }$ Then, we perform point-driven segmentation that use points as prompts in the BEV view to obtain the mask mBEV of the target object, as:

$$
m _ { B E V } = \mathrm { s e g } ( \mathrm { r e n d e r } ( c _ { B E V } ; \mathbf { G } ) , L _ { 2 D } ^ { B E V } )\tag{6}
$$

where $L _ { 2 D } ^ { B E V } ~ = ~ \mathcal { P } ( L _ { 3 D } , c _ { B E V } )$ denotes the 2D projection of the 3D point $L _ { 3 D }$ onto the BEV plane, and $\begin{array} { r l } { c _ { B E V } } & { { } = } \end{array}$ camera $( L _ { 3 D } ; \vec { u } , \bar { r } )$ denotes the camera pose that can be used for BEV rendering and $\{ \vec { u } , \bar { r } \}$ represents the up vector and the height. By applying Frustum Filtering in the BEV view, we can obtain a coarse grounding result.

Frustum Filtering (FF) utilizes a 2D mask to filter 3D Gaussians through a mask-shaped frustum. Specifically, we project the positions of all Gaussian primitives onto the plane of the 2D mask; those that fall outside the mask are considered background, while those within the mask are identified as target Gaussians. This Frustum Filtering (FF) on each Gaussian can be formulated as:

$$
\mathbf { I } _ { t a r } ^ { ( i ) } = \mathcal { F } ( \mathbf { G } ^ { ( i ) } ) = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f } p r o j e c t ( \mathbf { G } ^ { ( i ) } ; c _ { m } ) > 0 } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{7}
$$

where $\mathcal F ( \cdot )$ denotes FF, $\mathbf { I } _ { t a r } ^ { ( i ) } \in \mathbb { R } ^ { 1 }$ denotes the flag of the i-th Gaussian primitive that 1 denotes target and 0 denotes background, and $c _ { m }$ represents the camera parameters of plane of the 2D mask. With FF, we can quickly exclude Gaussian primitives that do not belong to the target object, thereby obtaining a coarse grounding result $\mathbf { G } ^ { ( \mathbf { I } _ { t a r } ) }$ as:

$$
\mathbf { G } ^ { ( \mathbf { I } _ { t a r } ) } = \mathcal { F } ( m _ { 2 D } ; \mathbf { G } , c _ { p l a n e } )\tag{8}
$$

where $\mathbf { G } ^ { ( \mathbf { I } _ { t a r } ) } \in \mathbb { R } ^ { S \times 5 9 }$ denotes the set of Gaussian primitives that are identified as targets, and S is the number of target Gaussians. $m _ { 2 D }$ denotes the 2D mask used for filtering.

<!-- image-->  
Fig. 1. Overall pipeline of our proposed GVR. (A) Preparation: Construct a Semantic Vector Book using SAM and CLIP, and obtain a Depth Book via 3DGS depth rendering. (B) Query: Encode the text query with CLIPâs text encoder, compute similarity with the multi-view semantic vector library, select top-k relevant views and object patches, and estimate the 3D object location via multi-view back-projection and stereo voting. (C) Segmentation: Perform point-driven segmentation on the BEV using the projected 3D location, obtain coarse target Gaussians via frustum filtering, and refine them through surrounding multi-view frustum intersection.

Surrounding Multi-view Frustum Intersection (SMFI) refines the grounding results. First, we generate k (typically $k = 4 )$ surrounding virtual cameras $\mathbf { C } _ { v i r }$ based on the 3D location $L _ { 3 D }$ . The distance $d _ { v i r }$ between each virtual camera and the target can be selected in the BEV, typically set to 3Ã the targetâs width. The pitch angle $\theta _ { v i r }$ of the virtual cameras is fixed at 30â¦. This step can be formulated as:

$$
\mathbf { C } _ { v i r } ^ { ( i ) } = C a m e r a ( \phi ^ { ( i ) } ; L _ { 3 D } , d _ { v i r } , \theta _ { v i r } , \vec { u } ) , i = 1 , 2 , \dots , k\tag{9}
$$

where $\phi ^ { ( i ) }$ denotes the yaw angle of the i-th virtual camera, which is uniformly sampled from [0, 360â¦), and âu represents the up vector of the camera. Then, we render views of the coarse target Gaussians $\mathbf { G } ^ { \left( \mathbf { I } _ { t a r } \right) }$ from each virtual camera, and feed these views into a text-driven segmenter (such as Grounded-SAM [14]). After obtaining the 2D mask for each view, we apply FF in each plane as:

$$
\tilde { \mathbf { G } } ^ { ( \mathbf { I } _ { t a r } ) ( i ) } = \mathcal { F } ( \mathrm { s e g } ( v _ { i } , Q ) ; \mathbf { G } ^ { ( \mathbf { I } _ { t a r } ) } , c _ { i } )\tag{10}
$$

$$
\mathbf { G } ^ { ( \mathbf { I } _ { t a r } ) } = \tilde { \mathbf { G } } ^ { ( \mathbf { I } _ { t a r } ) ( i ) } \cap \ldots \cap \tilde { \mathbf { G } } ^ { ( \mathbf { I } _ { t a r } ) ( k ) }\tag{11}
$$

where $\mathbf { G } ^ { ( \mathbf { I } _ { t a r } ) } \in \mathbb { R } ^ { S \times 5 9 }$ denotes the set of Gaussian primitives that are identified as the target, and S is the number of target Gaussians.

## 3. EXPERIMENTS

Dataset. We evaluate our method on two standard 3DVG benchmarks: LERF-Mask [16] and 3D-OVS [17]. The performance of GVR is evaluated using two main metrics: Localization Accuracy (Acc) and Intersection over Union (IoU). Implementation Details. Our method utilizes three visual foundation models: SAM2 [12], CLIP [13], and Grounding DINO [14]. In our experimental setting: for CLIP, we use OpenCLIP ViT-B/16 model; for SAM2, we use the ViT-H model; and for Grounding DINO, we use the DINO-SwinB model. The number of surrounding virtual cameras k is set to 4, the distance $d _ { v i r }$ is set to 3 times the targetâs width, and the pitch angle $\theta _ { v i r }$ is fixed at 30â¦. All experiments are conducted on a single NVIDIA RTX 4090 GPU with 24GB memory.

Quantitative analysis. As shown in Table 1, our GVR achieves the best overall results on both LERF-Mask and 3D-OVS datasets. Specifically, GVR outperforms previous SOTA methods in both Accuracy and IoU, achieving 87.5% Accuracy and 56.2% IoU on LERF-Mask, and 95.4% overall on 3D-OVS. These results demonstrate the effectiveness and generalization ability of our approach for zero-shot 3DVG.

Qualitative analysis. As shown in Fig. 2, 7 queries are conducted on 3 scenes from LERF-Mask and 3D-OVS. Unlike LangSplat, which often localizes only part of the object due to its pixel-wise lookup, GVR operates on Gaussians and produces more accurate and complete masks. Notably, since ReasonGrounder isnât open-sourced, our qualitative analysis only compares with the second-best method, LangSplat.

<!-- image-->  
Fig. 2. Qualitative comparison between LangSplat [11] and our GVR. Our method generates more precise and complete segmentation masks for the target objects described by the text queries. We visualize our results by dying Gaussians to red and splatting them on the camera plane.

Table 1. Quantitative comparison with baselines and ablation study on LERF-Mask and 3D-OVS. Metrics marked with â indicate higher is better. 1st 2nd 3rd denote the top three results. The ablation section reports the drop in performance compared to the full model.
<table><tr><td rowspan="2">Methods</td><td colspan="10">LERF-Mask</td><td colspan="6">3D-OVS</td></tr><tr><td colspan="2">Ramen Accâ mIoUâ Accâ mIoUâ|</td><td colspan="2">Figurines</td><td colspan="2">Teatime Accâ mIoUâ</td><td colspan="2">Kitchen Accâ mIoUâ</td><td colspan="2">Overall Accâ mIoUâ</td><td colspan="2">bed bench room sofa lawn overall</td><td colspan="2">All</td><td colspan="2"></td></tr><tr><td>LSeg [15]</td><td>14.1</td><td>7.0 8.9</td><td></td><td>7.6</td><td>33.9</td><td>21.7</td><td>27.3</td><td>29.9</td><td>| 21.1</td><td>16.6</td><td>56.0</td><td>6.0</td><td>19.2</td><td>4.5</td><td>17.5</td><td>20.6</td></tr><tr><td> L [16]</td><td>62.0</td><td>28.2</td><td>75.0</td><td>38.6</td><td>84.8</td><td>45.0</td><td>72.7</td><td>37.9</td><td>73.6</td><td>37.4</td><td>73.5</td><td>53.2</td><td>46.6</td><td>27</td><td>73.7</td><td>54.8</td></tr><tr><td>LangSplat [11]</td><td>73.2</td><td>51.2</td><td>80.4</td><td>44.7</td><td>88.1</td><td>65.1</td><td>95.5</td><td>44.5</td><td>84.3</td><td>51.4</td><td>92.5</td><td>94.2</td><td>94.1</td><td></td><td>90.0 96.1</td><td>93.4</td></tr><tr><td>ReasonGrounder [9]</td><td>78.5</td><td>53.4</td><td>82.4</td><td>49.6</td><td>89.7</td><td>68.2</td><td>96.2</td><td>49.3</td><td>86.7</td><td>55.1</td><td>93.3</td><td>96.6</td><td>94.5</td><td>91.7</td><td>97.3</td><td>94.7</td></tr><tr><td>Ours</td><td>79.0 54.1</td><td>85.3</td><td></td><td>51.2</td><td>90.0</td><td>69.5</td><td>95.8</td><td>50.1</td><td>| 87.5</td><td>56.2</td><td>93.1</td><td>96.6</td><td>95.3</td><td></td><td>94.3 97.8</td><td>95.4</td></tr><tr><td>w/o RFL</td><td>|-22.5 -18.2</td><td></td><td>| -18.3</td><td>-25.4</td><td>| -29.8</td><td>-37.9</td><td>-30.2</td><td>-26.1</td><td>|-25.2</td><td>-26.9</td><td></td><td></td><td>|-13.7 -18.4 -12.1</td><td>-21.2</td><td>-4.2</td><td>-13.9</td></tr><tr><td>w/o SMFI</td><td>-8.7</td><td>-13.5</td><td>-7.9</td><td>-12.6</td><td>-6.5</td><td>-14.2</td><td>-7.1</td><td>-13.8</td><td>-7.6</td><td>-13.5</td><td>-2.3</td><td>-6.7</td><td>-1.5</td><td>-2.3</td><td>-3.5</td><td>-3.3</td></tr><tr><td>w/o RFL &amp; SMFI</td><td>-23.6 -19.3</td><td></td><td>-20.7</td><td>-27.1</td><td>-30.8 -38.6</td><td></td><td>-33.2</td><td>-27.8</td><td>-26.6</td><td>-18.2</td><td></td><td></td><td>-15.9 -20.2 -13.4 -23.1 -5.1</td><td></td><td></td><td>-15.5</td></tr></table>

Table 2. Performance Comparison on Figurines scene. Preparation time means the time for a method ready for queries. Query speed means the time for a single query.
<table><tr><td>Method</td><td>Preparation Time</td><td>Query speed (s)</td></tr><tr><td>LangSplat</td><td>1h 30min</td><td>2.7</td></tr><tr><td>ReasonGrounder</td><td>48 min</td><td>0.6</td></tr><tr><td>GVR (Ours)</td><td>37s</td><td>0.25</td></tr></table>

Performance Analysis. As shown in Table 2, our GVR significantly reduces both preparation time and query speed compared to previous methods. Unlike prior approaches that require time-consuming per-scene training, GVR only needs to build the knowledge books, enabling real-time querying.

Ablation on RFL. We analyze the impact of the Retrieval for Localizing (RFL) module. If we do not use RFL, we directly use the text-driven segmentation on the BEV view to obtain the 2D mask for Frustum Filtering. As shown in Table 1, without RFL, the performance drops significantly (from 87.5 to 62.3 in Acc and from 56.2 to 29.3 in mIoU), indicating that RFL is crucial for accurately localizing the target object and providing reliable clues for subsequent segmentation.

Ablation on SMFI. Without SMFI, only the coarse segmentation from the BEV frustum filtering is used. As shown in Table 1, removing SMFI leads to a notable performance drop (from 87.5 to 79.9 in Acc and from 56.2 to 42.7 in mIoU), demonstrating that SMFI is effective in refining coarse results and improving the precision of target Gaussian segmentation.

## 4. CONCLUSIONS AND DISCUSSIONS

We introduce GVR, a novel zero-shot 3DVG framework for 3DGS. By reformulating the 3DVG task as a 2D retrieval problem, GVR effectively leverages existing 2D visual foundation models to achieve high-quality localization without the need for 3D annotated data or per-scene training. Our extensive experiments demonstrate that GVR not only achieves state-of-the-art performance on standard benchmarks but also significantly reduces training time and query latency.

## 5. REFERENCES

[1] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George Drettakis,Â¨ â3d gaussian splatting for real-time radiance field rendering.,â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[2] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke, âGaussian grouping: Segment and edit anything in 3d scenes,â in European Conference on Computer Vision. Springer, 2024, pp. 162â179.

[3] Seokhun Choi, Hyeonseop Song, Jaechul Kim, Taehyeong Kim, and Hoseok Do, âClick-gaussian: Interactive segmentation to any 3d gaussians,â in European Conference on Computer Vision. Springer, 2024, pp. 289â305.

[4] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping Liu, and Guosheng Lin, âGaussianeditor: Swift and controllable 3d editing with gaussian splatting,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 21476â21485.

[5] Jiazhong Cen, Jiemin Fang, Chen Yang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian, âSegment any 3d gaussians,â arXiv preprint arXiv:2312.00860, 2023.

[6] Xu Hu, Yuxi Wang, Lue Fan, Junsong Fan, Junran Peng, Zhen Lei, Qing Li, and Zhaoxiang Zhang, âSagd: Boundary-enhanced segment anything in 3d gaussian via gaussian decomposition,â arXiv preprint arXiv:2401.17857, 2024.

[7] Qiuhong Shen, Xingyi Yang, and Xinchao Wang, âFlashsplat: 2d to 3d gaussian splatting segmentation solved optimally,â in European Conference on Computer Vision. Springer, 2024, pp. 456â472.

[8] Runsong Zhu, Shi Qiu, Zhengzhe Liu, Ka-Hei Hui, Qianyi Wu, Pheng-Ann Heng, and Chi-Wing Fu, âRethinking end-to-end 2d to 3d scene segmentation in gaussian splatting,â arXiv preprint arXiv:2503.14029, 2025.

[9] Zhenyang Liu, Yikai Wang, Sixiao Zheng, Tongying Pan, Longfei Liang, Yanwei Fu, and Xiangyang Xue, âReasongrounder: Lvlm-guided hierarchical feature splatting for open-vocabulary 3d visual grounding and reasoning,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 3718â 3727.

[10] Yue Li, Qi Ma, Runyi Yang, Huapeng Li, Mengjiao Ma, Bin Ren, Nikola Popovic, Nicu Sebe, Ender Konukoglu, Theo Gevers, et al., âScenesplat: Gaussian splattingbased scene understanding with vision-language pretraining,â arXiv preprint arXiv:2503.18052, 2025.

[11] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister, âLangsplat: 3d language gaussian splatting,â in Proceedings of the IEEE/CVF Conference

on Computer Vision and Pattern Recognition, 2024, pp. 20051â20060.

[12] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al., âSegment anything,â in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4015â4026.

[13] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al., âLearning transferable visual models from natural language supervision,â in International conference on machine learning. PmLR, 2021, pp. 8748â8763.

[14] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Xuefei Ning, Zhiyang Dou, Zhiqiang Shen, Xiangyu Zhang, Yilun Chen, Yifei Huang, Yixuan Wei, Yanjie Han, Yutong Bai, Hongyang Li, Zehuan Yuan, and Jifeng Dai, âGrounding dino: Marrying dino with grounded pretraining for open-set object detection,â in ICCV, 2023.

[15] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and Rene Ranftl, âLanguage-driven semantic Â´ segmentation,â arXiv preprint arXiv:2201.03546, 2022.

[16] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik, âLerf: Language embedded radiance fields,â in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 19729â19739.

[17] Kunhao Liu, Fangneng Zhan, Jiahui Zhang, Muyu Xu, Yingchen Yu, Abdulmotaleb El Saddik, Christian Theobalt, Eric Xing, and Shijian Lu, âWeakly supervised 3d open-vocabulary segmentation,â Advances in Neural Information Processing Systems, vol. 36, pp. 53433â53456, 2023.