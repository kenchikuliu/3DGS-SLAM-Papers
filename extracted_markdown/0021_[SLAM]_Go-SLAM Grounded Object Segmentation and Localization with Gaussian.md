# Go-SLAM: Grounded Object Segmentation and Localization with Gaussian Splatting SLAM

Phu Pham1, Dipam Patel1, Damon Conover2, Aniket Bera1

1Department of Computer Science, Purdue University 2DEVCOM Army Research Laboratory {phupham, dipam, aniketbera}@purdue.edu, damon.m.conover.civ@army.mil

Abstractâ We introduce Go-SLAM, a novel framework that utilizes 3D Gaussian Splatting SLAM to reconstruct dynamic environments while embedding object-level information within the scene representations. This framework employs advanced object segmentation techniques, assigning a unique identifier to each Gaussian splat that corresponds to the object it represents. Consequently, our system facilitates open-vocabulary querying, allowing users to locate objects using natural language descriptions. Furthermore, the framework features an optimal path generation module that calculates efficient navigation paths for robots toward queried objects, considering obstacles and environmental uncertainties. Comprehensive evaluations in various scene settings demonstrate the effectiveness of our approach in delivering high-fidelity scene reconstructions, precise object segmentation, flexible object querying, and efficient robot path planning. This work represents an additional step forward in bridging the gap between 3D scene reconstruction, semantic object understanding, and real-time environment interactions.

## I. INTRODUCTION

Autonomous robots are becoming increasingly vital in various fields, including search and rescue, manufacturing, and military operations [1], [2], [3], [4]. To effectively navigate and interact with their environment, these robots need the ability to accurately reconstruct the surroundings, segment objects of interest, and plan paths in real-time. One of the key challenges in building such systems lies in achieving high-fidelity scene reconstruction while also integrating semantic understanding of objects within the scene. Additionally, enabling robots to query objects in an openvocabulary manner and generate optimal paths to interact with these objects enhances their flexibility and adaptability in challenging environments.

Traditional SLAM or Simultaneous Localization and Mapping techniques [5], [6], [7] have proven effective in reconstructing environments but often fail to provide detailed, object-level segmentation and interaction capabilities. In contrast, methods like point cloud or voxel-based reconstructions, while offering spatial accuracy, tend to struggle with incorporating object semantics in a robust and scalable manner. Recent advances in 3D Gaussian Splatting [8] offer a promising alternative for scene representation and rendering by using 3D Gaussian primitives to model the geometry and appearance of a scene.

While accurate 3D reconstruction is essential, true scene understanding requires the ability to identify and label objects within the environment. To address this, we incorporate advanced computer vision models that provide robust object detection and precise segmentation capabilities. By leveraging these techniques with 3D Gaussian Splatting, we generate a semantically rich environmental representation, where each Gaussian splat is associated with an object label. This enables robotic systems to understand both the spatial structure of the environment and the semantic relationships between objects, allowing for accurate object identification, tracking, and interaction across multiple camera frames.

<!-- image-->  
Fig. 1: The entire pipeline of Go-SLAM â The user queries the 3D reconstructed model in real-time with a specific object in the environment. The Go-SLAM detects the queried object and provides the 3D world coordinates of the goal location. The drone now navigates to the provided co-ordinates using PRM path planning algorithm

Another novelty of our approach is the support for openvocabulary queries. By incorporating natural language processing techniques, our system allows users or higher-level planning algorithms to locate objects using flexible, humanlike descriptions. This capability significantly enhances the adaptability of robotic systems, enabling them to understand and act upon a wide range of commands without being limited to a predefined set of object categories. Our entire approach is outlined in Figure 1.

Finally, we demonstrate the practical utility of our framework by implementing an optimal path-planning algorithm that leverages the semantically annotated 3D model. This allows a robot to efficiently navigate from its current position to a queried object, taking into account the spatial layout and potential obstacles in the environment.

The main contributions of this paper can be summarized as follows:

â¢ A novel implementation of 3D Gaussian Splatting SLAM with state-of the-art object segmentation and labeling techniques.

â¢ An open-vocabulary querying system that enables flexible object localization in 3D reconstructed environments.

â¢ Comprehensive experimental results demonstrate the effectiveness of our approach, showing improvements in precision, recall, and IoU by up to 17%, 27%, and 35%, respectively, across various scenarios.

By combining these components, our framework represents a significant step towards creating more intelligent and adaptable robotic systems. To the best of our knowledge, this is the first SLAM framework capable of both understanding and interacting with complex, unknown environments.

## II. RELATED WORK

## A. 3D Gaussian Splatting

3D Gaussian Splatting (3DGS) has emerged as a powerful technique for representing and rendering 3D scenes. Originally introduced by Kerbl et al.[8], 3DGS uses a collection of 3D Gaussian primitives to model geometry and appearance. This approach offers several advantages over traditional 3D reconstruction methods. Unlike meshbased [9], [10], which can be computationally expensive and complex to manipulate, Gaussian Splatting provides a more flexible representation by modeling surfaces with continuous, parameterized Gaussians. This allows for smoother surface approximations, especially for objects with irregular or complex geometries. Additionally, Gaussian Splatting can handle partial or noisy data more robustly, making it well-suited for real-time applications such as SLAM, where incomplete or uncertain information is common. The method also facilitates efficient integration of multimodal data, such as combining RGB and depth information, which enhances the accuracy of scene reconstructions.

## B. SLAM systems

Simultaneous Localization and Mapping (SLAM) is a fundamental problem in robotics and computer vision, aiming to construct a map of an unknown environment while simultaneously tracking an agentâs location within it. Traditional SLAM approaches have relied on sparse featurebased methods [11] or dense volumetric representations [12]. More recently, neural implicit representations like Neural Radiance Fields (NeRF) [13] have been adapted for tasks like SLAM [14] and navigation [15], offering high-quality scene reconstruction, but often at the cost of computational efficiency [16], [17], [14], [18].

## C. Gaussian Splatting SLAM

The integration of 3D Gaussian Splatting into SLAM systems is a recent development that aims to leverage the advantages of 3DGS for real-time mapping and localization. Matsuki et al. [19] introduced the first Gaussian Splatting SLAM system, demonstrating its effectiveness in monocular settings. GS-SLAM by Yan et al. [20] proposed a dense visual SLAM system using 3DGS, achieving competitive performance in both reconstruction and localization with lower time consumption compared to other methods. SplaTAM by

Keetha et al. [21] introduced a system for dense RGBD SLAM using 3DGS, demonstrating real-time performance and high-quality reconstruction.

These Gaussian Splatting SLAM approaches have shown promising results in terms of reconstruction quality, localization accuracy, and computational efficiency. However, challenges remain in areas such as large-scale mapping, loop closure, and handling dynamic environments.

## D. Object detection and segmentation

Object detection and segmentation are essential for enabling robots to understand and interact with their environment. Traditional approaches, such as Faster R-CNN [22] and Mask R-CNN [23] have been instrumental in detecting objects and generating instance-specific segmentation masks, enabling robots to recognize and differentiate objects within complex scenes. In addition, methods such as YOLO (You Only Look Once) [24] introduced single-shot detectors that achieve real-time object detection by predicting bounding boxes and class probabilities directly from full images, significantly improving the efficiency of object recognition tasks.

More recent models, such as Grounding DINO [25] and Segment Anything Model (SAM) [26], have pushed the boundaries of object segmentation. Grounding DINO uses transformer-based architectures to detect and localize objects by understanding semantic context, while SAM excels at universal segmentation by generating masks for any object given minimal input, without needing retraining. Grounded SAM, which integrates Grounding DINO with SAM, enables detection and segmentation of any objects specified by an input text prompt. These models empower robotic systems to recognize and segment objects flexibly, even in environments with undefined or unseen objects.

## E. Language embedded for 3D reconstruction

Recent developments in 3D scene reconstruction have increasingly incorporated language understanding, enabling more intuitive interactions with 3D environments. A notable work in this field is LERF (Language Embedded Radiance Fields) [27], which embeds CLIP-based language representations into Neural Radiance Fields (NeRF). LERF constructs a dense, multi-scale language field by rendering CLIP embeddings along training rays, facilitating zero-shot, pixel-aligned queries without the need for region proposals or masks. This approach supports real-time generation of 3D relevancy maps for diverse language prompts, offering potential applications in robotics, vision-language model analysis, and interactive scene exploration.

Building upon LERF, LangSplat [28] offers a more efficient approach for creating 3D language fields using 3D Gaussian Splatting rather than NeRF. By encoding CLIPbased language features [29] into 3D Gaussians, LangSplat significantly reduces computational cost through a tile-based splatting method. Additionally, it incorporates a language autoencoder to lower memory usage and uses SAM for hierarchical semantics learning, improving object boundary precision. These advancements lead to faster and more accurate open-vocabulary 3D object localization and semantic segmentation compared to LERF.

<!-- image-->  
Fig. 2: Overview of our Go-SLAM framework for environment reconstruction and language embedded feature.

## III. METHOD

In this section, we outline the comprehensive methodology of our framework, namely Go-SLAM, which employs cutting-edge techniques to achieve high-precision and efficient 3D reconstruction of environments captured through RGBD cameras. To the best of our knowledge, this is the pioneering SLAM system to integrate language features, enabling open-vocabulary object detection and localization. Figure. 2 illustrates the overview of our framework. In the following sections, we will elaborate on each component in the pipeline.

## A. 3D Gaussian Splatting SLAM framework

We employ 3D Gaussian Splatting to represent the reconstructed environment. Our proposed method builds upon the SplaTAM [21] approach to implement a robust and efficient 3DGS SLAM system. We leverage the strengths of explicit volumetric representations using 3D Gaussians to enable high-fidelity reconstruction from a single RGBD camera.

1) Differential 3D Gaussian Splatting representation: The core concept behind 3D Gaussian splatting is to represent the scene as a collection of Gaussians, where each Gaussian splat encodes key attributes like position, scale, orientation, color, opacity, and object association. These Gaussians act as probabilistic volumetric representations of the scene, allowing us to efficiently approximate the underlying geometry and appearance from multiple viewpoints.

Each splat is modeled as a 3D Gaussian distribution with parameters:

$$
G ( x ) = c \cdot \exp \left( - { \frac { 1 } { 2 } } ( x - \mu ) ^ { \top } { \Sigma } ^ { - 1 } ( x - \mu ) \right)\tag{1}
$$

where $ { \boldsymbol { \mathscr { x } } } ^ { \mathrm { ~ ~ } } \in ~ \mathbb { R } ^ { 3 }$ is a 3D point in space, $\mu \in \mathbb { R } ^ { 3 }$ is the Gaussian center, representing the position of the splat, $\Sigma \in \mathbb { R } ^ { 3 \times 3 }$ is the covariance matrix, defining the scale and orientation of the splat in 3D space, and $c \in \mathbb { R } ^ { 3 }$ is the color vector (RGB) associated with the splat,

The exponential term models the spatial influence of the splat, decaying with distance from the center $\mu .$ . Our framework uses this probabilistic approach to model both the geometry and appearance of the environment, with splats efficiently representing 3D surfaces across multiple viewpoints.

A key advantage of this 3DGS representation is that it allows for differentiable rendering, enabling us to optimize the parameters of each Gaussian (such as $\mu ,$ Î£, and c) through backpropagation. By rendering the scene from multiple viewpoints and comparing the rendered images with ground truth, we can compute a loss function, such as L1 or L2 loss, and backpropagate the error to adjust the parameters. This optimization process refines the Gaussian representations to better approximate the geometry and appearance of the scene, leveraging the differentiability of the rendering process to iteratively improve the 3D reconstruction.

2) Tracking and Gaussian densification: For each captured RGBD image, the framework back-projects each pixel (u, v) with depth d into 3D space using the cameraâs intrinsic matrix K, and converts it into a 3D Gaussian splat. The backprojection is computed as:

$$
X _ { c } = d \cdot K ^ { - 1 } \left[ \begin{array} { l } { u } \\ { v } \\ { 1 } \end{array} \right] , \quad X _ { w } = R _ { c } X _ { c } + t _ { c }\tag{2}
$$

where $X _ { c }$ and $X _ { w }$ are the point coordinates in camera and world frames, respectively, $R _ { c }$ and $t _ { c }$ are the rotation and translation matrices representing the cameraâs pose.

Each back-projected point is then converted into a Gaussian splat with its center $\mu$ set to $X _ { w } ,$ , and its color c taken from the corresponding pixel in the RGB image. We employ the camera tracking method introduced by [21] to estimate the camera pose for the current RGBD image. Camera parameters are optimized using gradient descent based on the L1 losses of the rendered colors and depths. Additionally, a silhouette mask is rendered to capture the density of the scene, which facilitates in quickly identifying previously mapped areas. This facilitates more efficient Gaussian densification for incoming RGBD images.

## B. Grounded object segmentation

In our framework, object segmentation plays a critical role in embedding semantic information into the reconstructed 3D environment. We use Grounding DINO [25] for object detection and SAM [26] for instance segmentation, as inspired by

<!-- image-->  
(a) YOLO + SAM

<!-- image-->  
(b) Faster RCNN + SAM

<!-- image-->  
(c) Grounding DINO + SAM  
Fig. 3: Performance comparison between different object detectors for grounded object segmentation.

Grounded SAM [30]. Grounding DINO combines visual and language features to detect objects by grounding text-based queries in specific image regions, requiring input labels as text prompts to identify objects. SAM (Segment Anything Model), on the other hand, generates high-quality instance segmentations without predicting object labels. Instead, it focuses purely on producing object masks, which can then be assigned labels through external means. This reliance on text-based inputs for Grounding DINO becomes problematic in unknown environments where predefined labels may not be available, limiting the systemâs autonomous capabilities.

Several existing models, such as Faster R-CNN [22], YOLO [24], and DETR [31], can directly provide object labels with bounding boxes. These models are highly effective in detecting objects and assigning labels based on the datasets they were trained on. However, a major drawback of these methods is their reliance on predefined class labels. They are typically trained on large datasets, such as COCO [32] or ImageNet [33], which contain a fixed set of object categories. As a result, these models perform poorly in open and unknown environments, where new or unseen object classes may appear.

To overcome these limitations, we integrate ChatGPT 4o model to autonomously analyzes captured images and generates a list of object labels without requiring predefined prompts or fixed object categories. By utilizing its strong visual understanding capability, the model identifies a wide range of objects, providing corresponding labels based on the visual content of the image. This allows for open-vocabulary detection, even in previously unseen environments.

Formally, given an image $I _ { i } ,$ the ChatGPT 4o model processes the image and outputs a list of object labels $\{ L _ { j } \}$ , where $L _ { j }$ corresponds to an object $O _ { j }$ detected in the image. These labels are then passed to Grounding DINO, which assigns bounding boxes $B _ { j }$ and classifies objects based on the provided labels. SAM is subsequently used to generate pixel-level segmentation masks $M _ { j }$

Given the segmentation mask, we assign unique object IDs to the corresponding 3D Gaussians in the scene. For each segmented object $O _ { j }$ , the pixels within the mask $M _ { j }$ are back-projected into 3D space, and the corresponding 3D Gaussian splats are updated with the object ID $C _ { j }$

For each object, $O _ { j }$ segmented in image $I _ { i } ,$ the object ID $C _ { j }$ is propagated to all Gaussians $G _ { k }$ corresponding to the pixels within the objectâs segmentation mask:

$$
C _ { j } = \{ G _ { k } \ | \ ( u _ { k } , v _ { k } ) \in M _ { j } \}\tag{3}
$$

This ensures that the semantic information from object segmentation is embedded directly into the 3D reconstruction, enabling effective querying and interaction with the scene.

To evaluate the performance of our proposed grounded object segmentation method, we conducted comparative experiments with two established object detectors, Faster R-CNN [22] and YOLO [24]. The results, illustrated in Figure 3, indicate that our method is capable of detecting nearly all objects in the scene with accurate segmentation. Conversely, both YOLO and Faster R-CNN struggle to recognize objects outside their pretrained categories, such as âguitarâ, âbackpackâ, and âcomputerâ. Moreover, these models misclassify âwhiteboardâ and âdesktop monitorâ as âtvâ, âwater bottleâ as âcupâ, and âsport bagâ as âsuitcaseâ. These findings demonstrate the robustness of our method in handling diverse and unknown environments.

## C. Open-vocabulary object queries

Our framework supports open-vocabulary object querying, enabling users to search for objects in the 3D reconstructed environment using textual descriptions. The process involves matching the input query with detected object classes and refining the search to ensure precise identification of the queried object. The overall pipeline is shown in Figure. 4.

1) Query matching: Given an input text query Q, we first compute the similarity between the CLIP embedding [29] of the input text and the CLIP embeddings of the detected object classes from the reconstructed environment. This allows us to determine the object class that most closely matches the input query based on semantic similarity.

For each detected object class $C _ { j }$ , we compute the cosine similarity between the CLIP embedding of the input query $E _ { Q }$ and the embedding of the object class $E _ { C _ { j } }$

$$
s = \cos ( E _ { Q } , E _ { C _ { j } } )\tag{4}
$$

The object class with the highest matching score s is selected as the best match for the query.

<!-- image-->  
Fig. 4: Open-vocabulary query pipeline

2) Pruning the search space: Once the most similar object class is identified, we prune the search space to include only the keyframes that contain instances of this detected object class. This reduces the search complexity by narrowing the candidate frames where the queried object may be located. However, the detected object classes are typically in their most general form (e.g., âtableâ instead of âdining tableâ or âchairâ instead of âswivel chairâ). To ensure precise object identification, we rerun the grounded segmentation on these keyframes, refining the object boundaries to better match the specific object characteristics from the query. If the queried object is not found in the selected keyframes, the system expands the search to the remaining keyframes that were initially excluded. This ensures that objects potentially missed in the initial pruning phase are still considered during the search process.

3) Object Localization: Once the object matching the query is identified, we backproject all its segmented pixels into 3D space and compute a 3D bounding box that encompasses the object. This bounding box provides an approximate spatial boundary of the object within the reconstructed environment.

Next, we select all the 3D Gaussian splats that lie within this bounding box. Each Gaussian splat has an associated object ID, and the set of object IDs corresponding to the Gaussians inside the bounding box is denoted as $C _ { \mathrm { i n } } .$ . Since the 3D bounding box may not fully capture the entire object, we expand the selection by including all Gaussians whose object IDs belong to $C _ { \mathrm { i n } }$ , ensuring that the complete object is included in the final set of Gaussians corresponding to the query Q:

$$
G _ { Q } = \{ G _ { k } \ | \ \mathrm { I D } ( G _ { k } ) \in C _ { \mathrm { i n } } \}\tag{5}
$$

For system evaluation and deployment, we utilized Gazebo (version 11.0) [34] in conjunction with ROS2 (Robot Operating System 2) [35] to simulate complex 3D environments. Our navigation strategy employs the PRM (Probabilistic Road Map) [36] path planning algorithm. This algorithm operates on a point cloud generated from the Gaussian centers derived from the reconstructed environment. The center of the objectâs bounding box, as determined by our object detection algorithm, serves as the goal point for path planning. This approach enables efficient obstacle avoidance and precise navigation within the reconstructed 3D space. By integrating these components, our system demonstrates robust performance in localizing queried objects and generating optimal paths for robot navigation, effectively bridging the gap between 3D scene reconstruction, semantic understanding, and real-time robotic interaction.

## IV. EXPERIMENTS AND RESULTS

This section details the experimental setup, evaluation metrics, and results obtained from testing our Go-SLAM framework on different scene settings. The experiments are structured to evaluate the robustness and accuracy of our system in varied environments.

## A. Experimental Setup

The experimental setup for assessing the effectiveness of our grounded object segmentation method included a variety of environments. For controlled testing, we utilized a subset of 18 scenes from the Replica dataset [37].

To establish a benchmark for our system, we conducted comparative evaluations against two baseline models. The first baseline utilized the Faster R-CNN segmentation model while maintaining the rest of our proposed pipeline. The second baseline incorporated our advanced grounded object segmentation coupled with 3D Gaussian Splatting SLAM (3DGS SLAM); however, it diverged from our full methodology by directly comparing the query against detected object labels, thereby omitting our carefully designed object matching algorithm. This approach allowed us to isolate the impact of our matching algorithm on the systemâs overall performance.

## B. Evaluation Metrics

The evaluation of model performance utilized precision and recall metrics, which assess the accuracy of identifying relevant objects and the capability to detect all human-labeled ground truth 3D bounding boxes, respectively. Additionally, the Intersection over Union (IoU) was measured, calculating the overlap between the predicted bounding boxes and these human-labeled ground truth bounding boxes, thereby providing a quantitative measure of localization accuracy.

<table><tr><td rowspan=2 colspan=1>Methods</td><td rowspan=1 colspan=3>Office 2</td><td rowspan=1 colspan=3>Room 0</td><td rowspan=1 colspan=3>Room 2</td></tr><tr><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>Recall</td><td rowspan=1 colspan=1>IoU</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>Recall</td><td rowspan=1 colspan=1>IoU</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>Recall</td><td rowspan=1 colspan=1>IoU</td></tr><tr><td rowspan=1 colspan=1>Baseline 1</td><td rowspan=1 colspan=1>0.38</td><td rowspan=1 colspan=1>0.42</td><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1>0.40</td><td rowspan=1 colspan=1>0.39</td><td rowspan=1 colspan=1>0.31</td><td rowspan=1 colspan=1>0.39</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=1>0.32</td></tr><tr><td rowspan=1 colspan=1>Baseline 2</td><td rowspan=1 colspan=1>0.54</td><td rowspan=1 colspan=1>0.58</td><td rowspan=1 colspan=1>0.44</td><td rowspan=1 colspan=1>0.52</td><td rowspan=1 colspan=1>0.56</td><td rowspan=1 colspan=1>0.42</td><td rowspan=1 colspan=1>0.53</td><td rowspan=1 colspan=1>0.57</td><td rowspan=1 colspan=1>0.43</td></tr><tr><td rowspan=1 colspan=1>Go-SLAM (ours)</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.59</td><td rowspan=1 colspan=1>0.60</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.56</td><td rowspan=1 colspan=1>0.62</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.58</td></tr></table>

TABLE I: Performance comparison of different methods across multiple environments.

## C. Results

The results of our experiments are evaluated both qualitatively and quantitatively to assess the effectiveness of our proposed grounded object segmentation method.

1) Qualitative results: Figure 5 shows the qualitative results of our 3D reconstruction of the âOffice 2â scene from the Replica dataset [37], comparing ground truth images with our rendered outputs. Using half of the original 2000 frames, our RGB and depth renderings closely match the ground truths, accurately capturing the sceneâs colors, object placement, and spatial depth, highlighting the precision of our reconstruction.

<!-- image-->  
(a) Ground truth RGB

<!-- image-->  
(b) Ground truth depth

<!-- image-->  
(c) Rendered RGB

<!-- image-->  
(d) Rendered depth  
Fig. 5: Reconstruction results of Office 2 scene from Replica dataset.

Figure 6 illustrates the 3D reconstruction of an âOffice 2â scene, highlighting the localization in response to the âOffice chairâ and âTabletâ queries. The queried objects, emphasized in red, are effectively identified from other elements within the scene, showcasing the capability of the system to identify queried objects accurately.

2) Quantitative results: Figure I displays the precision, recall, and Intersection over Union (IoU) metrics comparing our Go-SLAM method with two baseline approaches across three different environments: Office 2, Room 0, and Room 2. For each environment, the models were tasked with localizing 10 distinct objects within the scene.

In Office 2, Go-SLAM achieved a precision of 0.61, a recall of 0.73, and an IoU of 0.59, surpassing both Baseline 1 (0.38 precision, 0.42 recall, 0.33 IoU) and Baseline 2 (0.54 precision, 0.58 recall, 0.44 IoU). Similar trends are observed in Room 0 and Room 2, where Go-SLAM consistently outperforms the baselines in all metrics. The improvements over Baseline 1 demonstrate the robustness of our grounded object segmentation, particularly with challenging objects in natural settings. Additionally, the improvements over Baseline 2 highlight the effectiveness of our object localization techniques.

<!-- image-->  
(a) Segmented âoffice chairâ

<!-- image-->  
(b) Localized point cloud

<!-- image-->  
(c) Segmented âtabletâ

<!-- image-->  
(d) Localized point cloud  
Fig. 6: Visualization of the localized object with the queries âOffice chairâ and âTabletâ

## V. CONCLUSION

In conclusion, Go-SLAM introduces a novel approach to 3D scene reconstruction, combining Gaussian Splatting SLAM with state-of-the-art object segmentation and openvocabulary querying. Our framework successfully integrates 3D reconstruction, object detection, and natural language understanding to enable real-time environmental interactions. Through comprehensive experiments, we demonstrated that Go-SLAM achieves higher precision, recall, and IoU compared to the baseline methods, particularly in handling complex, unknown environments. The systemâs ability to seamlessly embed object-level information into the 3D scene allows for flexible object localization and querying. Overall, Go-SLAM represents a significant step forward in SLAM technology, bridging the gap between scene reconstruction and semantic object understanding.

## ACKNOWLEDGMENT

This material is based upon work supported in part by the DEVCOM Army Research Laboratory under cooperative agreement W911NF2020221.

[1] J. P. Queralta, J. Taipalmaa, B. Can Pullinen, V. K. Sarker, T. Nguyen Gia, H. Tenhunen, M. Gabbouj, J. Raitoharju, and T. Westerlund, âCollaborative multi-robot search and rescue: Planning, coordination, perception, and active vision,â IEEE Access, vol. 8, pp. 191 617â191 643, 2020.

[2] D. Drew, âMulti-agent systems for search and rescue applications,â Current Robotics Reports, vol. 2, 03 2021.

[3] A. Krnjaic, R. D. Steleac, J. D. Thomas, G. Papoudakis, L. Schafer, Â¨ A. W. K. To, K.-H. Lao, M. Cubuktepe, M. Haley, P. Borsting Â¨ et al., âScalable multi-agent reinforcement learning for warehouse logistics with robotic and human co-workers,â arXiv preprint arXiv:2212.11498, 2022.

[4] A. R. Cheraghi, S. Shahzad, and K. Graffi, âPast, present, and future of swarm robotics,â in Intelligent Systems and Applications, K. Arai, Ed. Cham: Springer International Publishing, 2022, pp. 190â233.

[5] C. Cadena, L. Carlone, H. Carrillo, Y. Latif, D. Scaramuzza, J. Neira, I. Reid, and J. J. Leonard, âPast, present, and future of simultaneous localization and mapping: Toward the robust-perception age,â IEEE Transactions on Robotics, vol. 32, no. 6, pp. 1309â1332, 2016.

[6] C. Campos, R. Elvira, J. J. G. RodrÂ´Ä±guez, J. M. M. Montiel, and J. D. Tardos, âOrb-slam3: An accurate open-source library for visual, Â´ visualâinertial, and multimap slam,â IEEE Transactions on Robotics, vol. 37, no. 6, pp. 1874â1890, 2021.

[7] Z. Teed and J. Deng, âDroid-slam: deep visual slam for monocular, stereo, and rgb-d cameras,â in Proceedings of the 35th International Conference on Neural Information Processing Systems, ser. NIPS â21. Red Hook, NY, USA: Curran Associates Inc., 2024.

[8] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering,â ACM Transactions on Graphics, vol. 42, no. 4, July 2023. [Online]. Available: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

[9] M. Kazhdan, M. Bolitho, and H. Hoppe, âPoisson surface reconstruction,â in Proceedings of the Fourth Eurographics Symposium on Geometry Processing, ser. SGP â06. Goslar, DEU: Eurographics Association, 2006, p. 61â70.

[10] W. E. Lorensen and H. E. Cline, âMarching cubes: A high resolution 3d surface construction algorithm,â in Proceedings of the 14th Annual Conference on Computer Graphics and Interactive Techniques, ser. SIGGRAPH â87. New York, NY, USA: Association for Computing Machinery, 1987, p. 163â169. [Online]. Available: https://doi.org/10.1145/37401.37422

[11] E. Rublee, V. Rabaud, K. Konolige, and G. Bradski, âOrb: An efficient alternative to sift or surf,â in 2011 International Conference on Computer Vision, 2011, pp. 2564â2571.

[12] R. A. Newcombe, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim, A. J. Davison, P. Kohi, J. Shotton, S. Hodges, and A. Fitzgibbon, âKinectfusion: Real-time dense surface mapping and tracking,â in 2011 10th IEEE International Symposium on Mixed and Augmented Reality, 2011, pp. 127â136.

[13] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â in ECCV, 2020.

[14] A. Rosinol, J. J. Leonard, and L. Carlone, âNerf-slam: Real-time dense monocular slam with neural radiance fields,â in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023, pp. 3437â3444.

[15] D. Patel, P. Pham, and A. Bera, âDronerf: Real-time multi-agent drone pose optimization for computing neural radiance fields,â in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2023, pp. 5050â5055.

[16] E. Sandstrom, Y. Li, L. Van Gool, and M. R. Oswald, âPoint- Â¨ slam: Dense neural point cloud-based slam,â in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[17] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys, âNice-slam: Neural implicit scalable encoding for slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[18] E. Sucar, S. Liu, J. Ortiz, and A. Davison, âiMAP: Implicit mapping and positioning in real-time,â in Proceedings of the International Conference on Computer Vision (ICCV), 2021.

[19] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, âGaussian splatting slam,â in Proceedings of the IEEE/CVF Conference on

Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 18 039â18 048.

[20] C. Yan, D. Qu, D. Xu, B. Zhao, Z. Wang, D. Wang, and X. Li, âGsslam: Dense visual slam with 3d gaussian splatting,â in CVPR, 2024.

[21] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, âSplatam: Splat, track & map 3d gaussians for dense rgb-d slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[22] S. Ren, K. He, R. Girshick, and J. Sun, âFaster r-cnn: towards realtime object detection with region proposal networks,â in Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 1, ser. NIPSâ15. Cambridge, MA, USA: MIT Press, 2015, p. 91â99.

[23] K. He, G. Gkioxari, P. Dollar, and R. Girshick, âMask r-cnn,â in Â´ 2017 IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2980â2988.

[24] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, âYou only look once: Unified, real-time object detection,â in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.

[25] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, C. Li, J. Yang, H. Su, J. Zhu et al., âGrounding dino: Marrying dino with grounded pre-training for open-set object detection,â arXiv preprint arXiv:2303.05499, 2023.

[26] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollar, and Â´ R. Girshick, âSegment anything,â in 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 3992â4003.

[27] J. Kerr, C. M. Kim, K. Goldberg, A. Kanazawa, and M. Tancik, âLerf: Language embedded radiance fields,â in International Conference on Computer Vision (ICCV), 2023.

[28] M. Qin, W. Li, J. Zhou, H. Wang, and H. Pfister, âLangsplat: 3d language gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 20 051â20 060.

[29] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, âLearning transferable visual models from natural language supervision,â in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, M. Meila and T. Zhang, Eds., vol. 139. PMLR, 18â24 Jul 2021, pp. 8748â8763. [Online]. Available: https://proceedings.mlr.press/v139/radford21a.html

[30] T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan, Z. Zeng, H. Zhang, F. Li, J. Yang, H. Li, Q. Jiang, and L. Zhang, âGrounded sam: Assembling open-world models for diverse visual tasks,â 2024.

[31] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, âEnd-to-end object detection with transformers,â in European conference on computer vision. Springer, 2020, pp. 213â 229.

[32] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick, âMicrosoft coco: Common objects in Â´ context,â in Computer Vision â ECCV 2014, D. Fleet, T. Pajdla, B. Schiele, and T. Tuytelaars, Eds. Cham: Springer International Publishing, 2014, pp. 740â755.

[33] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, âImagenet: A large-scale hierarchical image database,â in 2009 IEEE Conference on Computer Vision and Pattern Recognition, 2009, pp. 248â255.

[34] O. S. R. Foundation, âGazebo,â 2023, [Computer software]. [Online]. Available: http://gazebosim.org

[35] S. Macenski, T. Foote, B. Gerkey, C. Lalancette, and W. Woodall, âRobot operating system 2: Design, architecture, and uses in the wild,â Science Robotics, vol. 7, no. 66, p. eabm6074, 2022. [Online]. Available: https://www.science.org/doi/abs/10.1126/ scirobotics.abm6074

[36] L. Kavraki, P. Svestka, J.-C. Latombe, and M. Overmars, âProbabilistic roadmaps for path planning in high-dimensional configuration spaces,â IEEE Transactions on Robotics and Automation, vol. 12, no. 4, pp. 566â580, 1996.

[37] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma, A. Clarkson, M. Yan, B. Budge, Y. Yan, X. Pan, J. Yon, Y. Zou, K. Leon, N. Carter, J. Briales, T. Gillingham, E. Mueggler, L. Pesqueira, M. Savva, D. Batra, H. M. Strasdat, R. D. Nardi, M. Goesele, S. Lovegrove,

and R. Newcombe, âThe Replica dataset: A digital replica of indoor spaces,â arXiv preprint arXiv:1906.05797, 2019.