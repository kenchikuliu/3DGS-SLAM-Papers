# Beyond Averages: Open-Vocabulary 3D Scene Understanding with Gaussian Splatting and Bag of Embeddings

Abdalla Arafa1 and Didier Stricker2

1 abdalla.arafa@dfki.de

2 didier.stricker@dfki.de

Abstract. Novel view synthesis has seen significant advancements with 3D Gaussian Splatting (3DGS), enabling real-time photorealistic rendering. However, the inherent fuzziness of Gaussian Splatting presents challenges for 3D scene understanding, restricting its broader applications in AR/VR and robotics. While recent works attempt to learn semantics via 2D foundation model distillation, they inherit fundamental limitations: alpha blending averages semantics across objects, making 3D-level understanding impossible. We propose a paradigmshifting alternative that bypasses differentiable rendering for semantics entirely. Our key insight is to leverage predecomposed object-level Gaussians and represent each object through multiview CLIP feature aggregation, creating comprehensive âbags of embeddingsâ that holistically describe objects. This allows: (1) accurate open-vocabulary object retrieval by comparing text queries to object-level (not Gaussian-level) embeddings, and (2) seamless task adaptation: propagating object IDs to pixels for 2D segmentation or to Gaussians for 3D extraction. Experiments demonstrate that our method effectively overcomes the challenges of 3D open-vocabulary object extraction while remaining comparable to state-of-the-art performance in 2D open-vocabulary segmentation, ensuring minimal compromise.

Keywords: 3D Scene Understanding, Gaussian Splatting, 3D Open-Vocabulary Object Extraction, Open-Vocabulary Image Segmentation

## 1 Introduction

Novel view synthesis has gained significant popularity since the introduction of NeRF[9] (Neural Radiance Fields) in 2020, revolutionizing the field with its ability to render photorealistic scenes from novel viewpoints using neural networks. While NeRF pioneered neural scene representations with implicit volumetric rendering, recent advances such as 3D Gaussian Splatting[5] (3DGS) have shifted the paradigm by introducing efficient, explicit representations. By modeling scenes as collections of anisotropic 3D Gaussians, 3DGS achieves high-fidelity real-time rendering, unlocking new possibilities for dynamic scene reconstruction[20,3,19,4], photo-realistic human avatars[13,14], and text-to-3D generation, where 3DGS serves as a differentiable intermediate representation for 2D diffusion models[22,1,12].

3D Gaussian Splatting achieves remarkable performance in novel view synthesis, generating photorealistic renderings in real-time. However, its fuzzy, unstructured nature presents a key limitation: the inability to clearly distinguish or segment individual objects within the scene. Unlike explicit mesh-based representations, the splatted Gaussians lack inherent semantic or geometric boundaries, making tasks such as object-level editing, scene understanding, or compositional manipulation challenging. To address these limitations, recent works[15,17,23] have proposed distilling semantic knowledge from 2D foundation models such as CLIP[16] and DINOv2[11] with the help of extracted masks from SAM[7] into 3D Gaussians using differentiable rendering. Although these methods[15,17,23] correctly aggregate semantics in 2D image space - allowing open-vocabulary segmentation for rendered viewsâthey struggle to learn clean object-level semantics at the individual

Gaussian level. The noisy, inconsistent semantic attributes learned by the 3D Gaussians prevent reliable 3D scene understanding. Consequently, such approaches are limited to 2D search tasks (e.g., segmenting objects in a rendered frame) and fail to support 3D semantic queries (e.g., retrieving all instances of a specific object in the scene).

To overcome these limitations, we propose a fundamentally different approach that completely bypasses differentiable rendering for semantic learning. Instead of distilling noisy 2D semantics into individual Gaussians, we use a pre-trained 3D Gaussian scene (decomposed into distinct objects via Gaussian Grouping[21]) and extract multiview CLIP embeddings for each object, forming a comprehensive âbag of embeddingsâ that fully represents its visual appearance. For open-vocabulary querying, we perform retrieval directly at the object level: given a text prompt, we match it against the extracted embedding bags to identify relevant objects, then propagate their IDs to downstream tasks enabling precise 2D segmentation (filtering pixels by object ID) or 3D extraction (selecting Gaussians by ID). Our method decouples rendering from semantics, avoiding the pitfalls of alpha-blended semantic dilution while maintaining compatibility with real-time Gaussian splatting pipelines.

The contributions of this work are summarized as follows:

â Object-level Semantic Encoding: We propose to represent 3D Gaussian scenes as bags of multiview CLIP embeddings per object, eliminating the need for error-prone differentiable rendering of semantics.

â Open-Vocabulary 3D Search by Design: By operating on pre-grouped multiview scenes, our approach enables direct object-level retrieval for text queries, resolving the inconsistency of per-Gaussian semantics in prior work.

Task-Agnostic Semantic Propagation: We demonstrate how object IDs can be seamlessly propagated to both 2D segmentation (pixel-level filtering) and 3D extraction (Gaussian selection), unifying open-vocabulary tasks without retraining.

## 2 Related Work

## 2.1 Neural Rendering

Neural rendering has emerged as a powerful approach to synthesizing photorealistic images by combining the principles of computer graphics with deep learning. Early work, such as Neural Radiance Fields (NeRF)[9], introduced implicit volumetric representations, enabling novel view synthesis of complex 3D scenes. Subsequent improvements have accelerated training[10].

In addition to implicit representations, explicit approaches such as Gaussian splatting[5] have gained prominence. These methods use point-based representations with Gaussian kernels, offering efficient rendering and enhanced scalability for real-time applications. Such explicit methods are particularly useful in scenarios requiring interactive performance without sacrificing quality.

Using both implicit and explicit representations, neural rendering has significantly expanded the possibilities for rendering and understanding 3D scenes, making it a critical tool for applications in virtual reality, robotics, and beyond.

## 2.2 Open Vocabulary Scene Understanding

Recent advances in open-vocabulary scene understanding aim to integrate 2D vision-language foundation models (VLMs) with 3D scene representations, combining the semantic richness of models like CLIP[16] and SAM[7] with the geometric fidelity of Neural Radiance Fields (NeRF)[9] and 3D Gaussian Splatting (3DGS)[5]. Language Embedded Radiance Fields (LERF)[6] pioneered this integration by embedding CLIP features into NeRF, enabling spatial queries such as locating objects described in natural language (e.g., âblue mugâ). However, LERFâs reliance on volumetric rendering incurs significant computational costs and suffers from semantic inconsistencies due to varying CLIP embeddings across viewpoints. CLIP-NeRF[18] extends these capabilities to enable object editing but inherits the limitations of scalability and rendering efficiency inherent to NeRF.

The emergence of 3DGS as a rasterization-based alternative to NeRF has enabled faster, more efficient fusion of VLMs with 3D scene representations. LangSplat[15] uses CLIP embeddings and a scene-level autoencoder to achieve real-time open-vocabulary querying while resolving ambiguities like distinguishing multiple instances of the same object. Gaussian Grouping[21] combines SAMâs zero-shot segmentation with multi-view mask aggregation to achieve open-vocabulary instance segmentation, enabling object-level grouping without predefined categories. Feature3DGS[23] distills features from 2D models like DINOv2[11] or SAM into 3D Gaussians, facilitating semantic segmentation and high-quality rendering, although its static features limit dynamic language-driven interactions. LeGaussians[17] further pushes the boundary by jointly optimizing 3D Gaussians with CLIP embeddings through feature reconstruction losses, balancing semantic and geometric fidelity but facing challenges with feature drift due to viewpoint inconsistencies.

While these methods show significant progress, they mainly focus on pixel-level open-vocabulary understanding, which limits their ability to fully capture the semantics of 3D scenes. By relying on per-pixel features or aggregated masks, these approaches often struggle to distinguish individual objects or entities in 3D space. This makes it difficult to handle tasks that require explicit object-level reasoning, such as understanding spatial relationships or interactions in 3D environments. Overcoming this limitation is essential for advancing open-vocabulary scene understanding and enabling its use in applications like robotics, autonomous systems, and immersive virtual environments.

## 3 Method

In this work, we propose a novel approach to 3D scene understanding that facilitates object search in both image space and world space using open-vocabulary text queries. While previous methods[15,17,23] have achieved notable success in embedding language features into Gaussian Splatting via differentiable rendering for open-vocabulary segmentation in image space, they face significant challenges in 3D. Specifically, the reliance on alpha blending to learn semantics limits their ability to accurately capture and represent 3D semantics, hindering tasks such as object selection or extraction based on learned 3D features.

To address these limitations, our approach moves away from training individual Gaussians or pixels for language feature representation. Instead, we focus on grouping 3D Gaussians into distinct objects and associating language features with these abstract object groups rather than individual Gaussians. This shift enables open-vocabulary searches to operate at the object ID level, significantly simplifying tasks like 3D object extraction and 2D segmentation, as they reduce to filtering objects based on their assigned IDs.

The rest of this section is structured as follows: we first introduce the foundational concepts of the Gaussian Grouping[21] method, which forms the basis of our approach. Next, we discuss our Bag of Features Embedding strategy for representing grouped objects. Finally, we detail how we implement open-vocabulary text queries to support object search in both 2D and 3D spaces.

## 3.1 Preliminaries: Gaussian Grouping

Gaussian Grouping[21] introduces a method to enable fine-grained 3D scene understanding by assigning instance or stuff IDs to 3D Gaussians. This is achieved by augmenting Gaussians with a learnable Identity Encoding parameter, a compact vector of length 16, which represents unique object identities across the scene.

Input Preparation We first generate 2D masks from multi-view images using the SAM model[7]. Each mask is assigned a unique ID across views by treating the multi-view setup as a video sequence and employing a zero-shot tracker[2] for mask consistency. The output from this stage is a consistent pixel-aligned object ID for each object in the image.

3D Gaussian Rendering and Identity Encoding The Identity Encoding is optimized using a differentiable 3D Gaussian renderer. This renderer projects 3D Gaussians into 2D mask identity features by blending Identity Encoding vectors with Gaussian alpha contributions resulting in an output $I \doteq R ^ { 1 6 \times H \times \bigcirc }$ . Finally, a linear layer is applied followed by a softmax activation function to the rendered Identity Encoding of each pixel in the 2D image. This transforms the encoded features into a probability distribution, indicating the likelihood of each pixel belonging to a specific object ID in the scene.

Grouping Loss To supervise identity learning, Gaussian Grouping combines two losses:

â 2D Identity Loss: Uses cross-entropy classification over the rendered 2D mask identities, ensuring alignment with the 2D input labels.

â 3D Regularization Loss: Encourages spatial consistency by minimizing feature distances between neighboring Gaussians in 3D space. This ensures robust identity learning even for occluded or sparsely visible objects.

<!-- image-->  
Fig. 1: Multiview Bag of Features Embedding Extraction. Given multiple views pf a masked 3D object, we extract the visual embeddings of each view using CLIP image encoder. These multiview embeddings are aggregated into a bag-of-embeddings representation, which captures the semantic content of the object from different perspectives. This bag-of-embeddings representation enables open-vocabulary 2D segmentation and 3D object selection using text queries.

## 3.2 Multiview Bag of Features Embedding

To enable open-vocabulary queries about fine-grained object attributes, we augment each 3D object with a bag of CLIP embeddings aggregated across all its 2D appearances. While a straightforward approach would involve averaging embeddings across views to create a unified representation, this method risks discarding semantic details visible only in specific perspectives, such as a book title visible only from a close-up view or a logo visible exclusively in frontal views. By preserving the full set of embeddings, our method retains these rare but distinctive characteristics, allowing language queries to target both prominent and subtle attributes.

The extraction of language embeddings operates independently from the training of Gaussian splatting with group IDs, which allows performing these two steps in parallel. The process begins with the input data prepared for Gaussian grouping [21]. For each object ID, the corresponding object is masked across all training views to isolate it from the background and other objects. As illustrated in Figure 1, the masked views are passed through a CLIP encoder to extract semantic features for each view. These features are then aggregated into a bag of embeddings, enabling openvocabulary searches for relevant object IDs based on text queries.

This strategy takes advantage of CLIPâs ability to retrieve fine-grained details. Since each embedding represents a specific view of the object, the collection of embeddings captures the full diversity of the object. During a query, this approach can match not only the common features of the object but also the unique details visible from specific angles. This ensures that the representation of an object is not limited to its âaverageâ appearance but reflects all its visual traits, even those that appear infrequently across views.

<!-- image-->  
Fig. 2: Overview of our open-vocabulary 3D object selection and 2D segmentation pipeline. (a) Multiview image embeddings of 3D objects are matched with a text query (e.g., âJake from adventure timeâ) to extract relevant object IDs. (b) Using the retrieved object IDs, we filter the 3D Gaussians to render only the relevant object, enabling 3D object selection. (c) To perform open-vocabulary 2D segmentation, the full set of Gaussians is rendered from a specific camera view to produce a photorealistic image and an accompanying segmentation map. The object IDs obtained in (a) are then used to generate a binary mask that localizes the queried object in the image, enabling text-driven segmentation directly in rendered views.

## 3.3 Open-vocabulary Search

As shown in Fig. 2(a), we perform a text query search in the object embedding space to identify the object IDs most relevant to an open-vocabulary input text query. The task is formulated as follows: Given an object i represented by a bag of embeddings $\phi _ { o b j e c t } ^ { i } ,$ which are extracted following the process illustrated in Fig. 1, we compute the relevancy $\rho ^ { i , j }$ between each embedding $\phi _ { o b j e c t } ^ { i , j }$ and a text query $\phi _ { q u e r y }$ . The relevancy score is calculated using the approach described in LERF [6] and is defined as

$$
\rho ^ { i , j } = \operatorname* { m i n } _ { k } \frac { \exp ( \phi _ { o b j e c t } ^ { i , j } \cdot \phi _ { q u e r y } ) } { \exp ( \phi _ { o b j e c t } ^ { i , j } \cdot \phi _ { q u e r y } ) + \exp ( \phi _ { o b j e c t } ^ { i , j } \cdot \phi _ { c a n o n } ^ { k } ) }
$$

Here, $\phi _ { c a n o n } ^ { k }$ represents the CLIP embeddings of predefined general canonical queries such as âobjectâ, âstuffâ, and âtextureâ. This formulation computes the relevancy score between a single embedding of an object and the text query, effectively measuring how well the embedding aligns with the given query in the presence of canonical concepts.

To compute the overall relevancy score for an object, we get the relevancy scores of all its embeddings with the input query. The final score is derived by averaging the top k relevancy scores, where k is a hyperparameter that balances capturing fine-grained semantic details and ensuring accurate object identification. This process is mathematically expressed as:

$$
\rho ^ { i } = \frac { 1 } { k } \sum _ { j = 1 } ^ { k } \rho ^ { i , [ j ] }
$$

where $\rho ^ { i , \left[ j \right] }$ denotes the j-th highest relevancy score among all embeddings of object i.

With the object relevancy scores computed, searching in either image space or world space can be performed in a systematic manner. The object IDs with the highest relevancy scores are identified and their associated representations are extracted. For world-space object extraction, Gaussians corresponding to the selected object IDs are retrieved. For open-vocabulary segmentation in the image plane, pixels with matching IDs are masked to segment the relevant objects.

After extracting object IDs relevant to a text query, performing downstream tasks such as image segmentation or 3D object selection can be efficiently performed. As illustrated in Fig.2, filtering by object IDs is integrated at different stages of the rasterization process to accomplish the desired task. In panel b), the Gaussians are filtered based on their object IDs before rasterization, enabling 3D object selection. In panel c), the Gaussians are first rendered from the desired viewpoint, producing an RGB image along with a segmented image containing pixel-aligned object IDs. A mask is then generated from the segmented image using the extracted object IDs from the text query. Finally, this mask is applied to the rendered image to isolate and segment the desired object.

## 4 Experiments

## 4.1 Setup

Datasets We evaluate our approach using two distinct datasets. The first dataset, LERF[6], contains various real-world scenes captured with a standard iPhone camera, providing a rich set of in-the-wild scenarios. The second dataset, 3D-OVS[8], features a wide range of object categories, including those with long-tail distributions. To evaluate 2D mask segmentation on the 3D-OVS dataset, we follow the evaluation protocol specified in [8]. While for LERF dataset, we follow the evaluation protocol outlined in [15].

Metrics We evaluate the performance of our method on both 2D and 3D semantic segmentation tasks. For novel view synthesis of 2D masks, we use the mean Intersection over Union (mIoU) and localization accuracy. For 3D object extraction, we present a qualitative and visual evaluation of our method compared to the state-of-the-art.

Implementation Details We implement our method based on Gaussian Grouping[21] with the same hyperparameters provided in the paper with an identity encoding of 16 dimensions for each Gaussian. Moreover, during open-vocabulary search queries, we use k = 5 for averaging the top relevancy score and we employ a postprocessing step by ignoring objects that were not visible in at least 20% of the images in the dataset to remove the noisy output coming from DEVA[2].

<!-- image-->  
Fig. 3: A qualitative comparison between our method and Langsplat[15] on open-vocabulary 3D object extraction on the LERF dataset[6]. Given the same text queries, our method produces significantly more accurate object reconstructions, with the extracted Gaussians preserving both the geometry and appearance of the target object. In contrast, LangSplat often fails to recover complete objectsâfrequently producing fragmented or partial results, with large portions of the target object missing. This highlights the effectiveness of our approach in achieving precise and complete 3D object selection from text.

## 4.2 Open Vocabulary 3D Object Selection

Our primary focus in this work is on advancing 3D object extraction using open-vocabulary text queries. Traditional methods for embedding semantics into Gaussians often depend on explicit language training, which can be prone to noise, especially when using differentiable rendering. In contrast, our approach addresses this limitation by leveraging the strengths of Gaussian Grouping [21] and multiview semantic embeddings on the object level, enabling robust and flexible openvocabulary 3D object extraction.

The process illustrated in Fig.2b begins with a text query, which retrieves the most relevant object IDs based on semantic similarity. Once these IDs are identified, the Gaussians associated with them are extracted, facilitating a structured and meaningful reconstruction of the objects in 3D space. By operating directly on semantic representations, our method bypasses the challenges of explicit language training, ensuring greater accuracy and resilience to noise.

To evaluate the effectiveness of our approach, we performed a qualitative comparison against state-of-the-art methods, as shown in Fig.3. Our method demonstrates the ability to extract Gaussians representing objects solely based on input text queries, while preserving both the geometry and appearance of the objects. For instance, in the âgreen appleâ text query, our approach reconstructs the complete geometry of the object from the desired view, even when it is largely occluded in the original view. This highlights the advantages of 3D object extraction in uncovering occluded structures and ensuring a comprehensive representation. In contrast, LangSplat [15], while effective for 2D image segmentation, struggles with 3D text queries, often producing noisy or incomplete outputs for the desired object. These comparisons underscore the robustness and versatility of our method in open-vocabulary 3D object extraction.

## 4.3 Open Vocabulary Querying in 2D

Although the primary objective of this work is 3D object extraction, we demonstrate that our approach also delivers competitive performance in open-vocabulary 2D segmentation, aligning with state-of-the-art methods and even surpassing them in certain cases. This underscores the versatility and robustness of our method in addressing both 3D and 2D tasks.

<table><tr><td>Methods</td><td>figurines</td><td>ramen</td><td>teatime</td><td>overall</td></tr><tr><td>LERF[6]</td><td>28.2</td><td>37.9</td><td>38.6</td><td>37.3</td></tr><tr><td>Langsplat[15]</td><td>44.7</td><td>51.2</td><td>65.1</td><td>53.7</td></tr><tr><td>Gaussian Grouping[21]</td><td>69.7</td><td>45.8</td><td>61.9</td><td>59.1</td></tr><tr><td>Ours</td><td>63.3</td><td>53.7</td><td>79.3</td><td>65.4</td></tr></table>

Table 1: IOU performance on LERF dataset[6]. The best performance per column across all methods is indicated in bold.

<table><tr><td>Methods</td><td>figurines</td><td>ramen</td><td>teatime</td><td>overall</td></tr><tr><td>LERF[6]</td><td>75.0</td><td>62.0</td><td>84.8</td><td>58.1</td></tr><tr><td>Langsplat[15]</td><td>80.4</td><td>73.2</td><td>88.1</td><td>80.6</td></tr><tr><td>Ours</td><td>87.5</td><td>74.7</td><td>100</td><td>87.4</td></tr></table>

Table 2: Localization Accuracy performance on LERF dataset[6]. The best performance per column across all methods is indicated in bold.

The experimental setup for open-vocabulary 2D segmentation, illustrated in Fig.2c), involves rendering 3D Gaussians into a specific 2D view. Each Gaussian is rendered along with its identity encoding, producing a pixel-wise categorization of object IDs. Using an input text query, we identify the objects most semantically relevant to the query. These relevant object IDs are then used to segment the 2D view by isolating the pixels corresponding to these IDs.

<table><tr><td>Methods</td><td>bed</td><td>bench</td><td>room</td><td>lawn</td><td>sofa</td><td>overall</td></tr><tr><td>LERF[6]</td><td>88.9</td><td>71.4</td><td>66.1</td><td>81.2</td><td>77.5</td><td>77</td></tr><tr><td>3D-OVS[8]</td><td>89.5</td><td>89.3</td><td>92.8</td><td>88.2</td><td>74</td><td>86.8</td></tr><tr><td>LEGaussians[17]</td><td>45.7</td><td>47.4</td><td>44.7</td><td>49.7</td><td>48.2</td><td>47.1</td></tr><tr><td>Langsplat[15]</td><td>92.5</td><td>94.2</td><td>94.1</td><td>96.1</td><td>90</td><td>93.4</td></tr><tr><td>Ours</td><td>98</td><td>96.1</td><td>89.7</td><td>95</td><td>78.2</td><td>91.4</td></tr></table>

Table 3: IOU performance on 3D-OVS dataset[8]. Best values are highlighted in green , while second best values are highlighted in orange .

As shown in Tables 1 and 2, our model, evaluated on the LERF dataset [6], surpasses the stateof-the-art methods, including LangSplat [15], across both metrics, demonstrating the effectiveness of our multiview bag of embeddings representation. Additionally, Table 3 presents our performance on the 3D-OVS dataset [8], where our method outperforms the state-of-the-art on two individual datasets and ranks second overall. The slight drop in overall performance is attributed to certain queries in this dataset, which involve segmenting background regions. These scenarios pose challenges for the DEVA model [2], which sometimes struggles to fully capture background semantics, making it more difficult for the Gaussians to accurately learn the background identity.

## 5 Conclusion

In this work, we introduced a novel approach to 3D scene understanding that addresses the limitations of existing methods in open-vocabulary object retrieval and segmentation. By utilizing predecomposed object-level Gaussians from Gaussian Grouping and representing each object with a robust bag of multiview CLIP embeddings, our method achieves accurate and reliable open-vocabulary search in both 2D and 3D spaces. Unlike prior approaches that depend on differentiable rendering for semantic learningâoften resulting in noisy and inconsistent per-Gaussian featuresâour method circumvents these challenges by directly operating on object-level semantics, ensuring a cleaner and more coherent representation.

Our experimental results demonstrate that the proposed method excels in 3D object extraction while achieving competitive or superior performance in open-vocabulary 2D segmentation. The ability to seamlessly propagate object-level semantic queries to both pixel-level segmentation and 3D Gaussian selection underscores the flexibility and adaptability of our framework. Notably, the method effectively reconstructs occluded structures and supports fine-grained 3D object selection, further solidifying its utility across diverse applications.

Future directions for this research include integrating advanced vision-language models to enhance semantic understanding, improving the robustness of object grouping in complex or cluttered scenes, and extending the framework to dynamic scenes with temporal consistency.

Ultimately, our method represents a significant step forward in bridging the gap between highfidelity rendering and structured 3D scene understanding. By enabling robust open-vocabulary search and segmentation in 2D and 3D, it opens new avenues for applications in AR/VR, robotics, and interactive 3D editing, paving the way for more intuitive and scalable scene comprehension.

## References

1. Zilong Chen, Feng Wang, Yikai Wang, and Huaping Liu. Text-to-3d using gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21401â 21412, 2024.

2. Ho Kei Cheng, Seoung Wug Oh, Brian Price, Alexander Schwing, and Joon-Young Lee. Tracking anything with decoupled video segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1316â1326, 2023.

3. Yuanxing Duan, Fangyin Wei, Qiyu Dai, Yuhang He, Wenzheng Chen, and Baoquan Chen. 4d gaussian splatting: Towards efficient novel view synthesis for dynamic scenes. arXiv preprint arXiv:2402.03307, 2024.

4. Tobias Fischer, Jonas Kulhanek, Samuel Rota Bul\`o, Lorenzo Porzi, Marc Pollefeys, and Peter Kontschieder. Dynamic 3d gaussian fields for urban areas. arXiv preprint arXiv:2406.03175, 2024.

5. Bernhard Kerbl, Georgios Kopanas, Thomas LeimkÂ¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023.

6. Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. Lerf: Language embedded radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 19729â19739, October 2023.

7. Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015â4026, 2023.

8. Kunhao Liu, Fangneng Zhan, Jiahui Zhang, Muyu Xu, Yingchen Yu, Abdulmotaleb El Saddik, Christian Theobalt, Eric Xing, and Shijian Lu. Weakly supervised 3d open-vocabulary segmentation. Advances in Neural Information Processing Systems, 36:53433â53456, 2023.

9. Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021.

10. Thomas MÂ¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):1â15, 2022.

11. Maxime Oquab, TimothÂ´ee Darcet, ThÂ´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

12. Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.

13. Shenhan Qian, Tobias Kirschstein, Liam Schoneveld, Davide Davoli, Simon Giebenhain, and Matthias NieÃner. Gaussianavatars: Photorealistic head avatars with rigged 3d gaussians. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20299â20309, 2024.

14. Zhiyin Qian, Shaofei Wang, Marko Mihajlovic, Andreas Geiger, and Siyu Tang. 3dgs-avatar: Animatable avatars via deformable 3d gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5020â5030, 2024.

15. Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d language gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20051â20060, 2024.

16. Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748â8763. PmLR, 2021.

17. Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-Hua Guan. Language embedded 3d gaussians for open-vocabulary scene understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5333â5343, 2024.

18. Can Wang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao. Clip-nerf: Text-and-image driven manipulation of neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3835â3844, 2022.

19. Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20310â20320, 2024.

20. Zeyu Yang, Hongye Yang, Zijie Pan, Xiatian Zhu, and Li Zhang. Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. arXiv preprint arXiv:2310.10642, 2023.

21. Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke. Gaussian grouping: Segment and edit anything in 3d scenes. In European Conference on Computer Vision, pages 162â179. Springer, 2024.

22. Taoran Yi, Jiemin Fang, Junjie Wang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, and Xinggang Wang. Gaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6796â6807, 2024.

23. Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21676â21685, 2024.