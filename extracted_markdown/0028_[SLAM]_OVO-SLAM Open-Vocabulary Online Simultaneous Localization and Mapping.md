# Open-Vocabulary Online Semantic Mapping for SLAM

Tomas Berriel Martins University of Zaragoza

Martin R. Oswald University of Amsterdam

Javier Civera University of Zaragoza

Abstractâ This paper presents an Open-Vocabulary Online 3D semantic mapping pipeline, that we denote by its acronym OVO. Given a sequence of posed RGB-D frames, we detect and track 3D segments, which we describe using CLIP vectors. These are computed from the viewpoints where they are observed by a novel CLIP merging method. Notably, our OVO has a significantly lower computational and memory footprint than offline baselines, while also showing better segmentation metrics than offline and online ones. Along with superior segmentation performance, we also show experimental results of our mapping contributions integrated with two different full SLAM backbones (Gaussian-SLAM and ORB-SLAM2), being the first ones using a neural network to merge CLIP descriptors and demonstrating end-to-end open-vocabulary online 3D mapping with loop closure.

## I. INTRODUCTION

Semantic mapping targets the estimation of the category to which each element in a scene belongs, along with a consistent geometric representation. Rich semantic representations in 3D are essential for advanced robotic applications. Traditionally, semantic 3D reconstruction has relied on a closed-set approach in both offline [1, 2] and online [3, 4] settings, including integrations into semantic Simultaneous Localization and Mapping (SLAM) systems [5â7]. However, these methods are constrained by a predefined set of categories, which limits their flexibility and applicability in openended, real-world environments.

Following the emergence of Contrastive Language-Image Pre-training (CLIP) [8], there has been a surge of interest in open-vocabulary 3D representations [9â11], including efforts in online mapping [12â14]âthough not yet in full SLAM systems. While these recent approaches have shown strong performance, their dependence on offline processing or ground-truth camera poses for mapping significantly limits their applicability in robotics, augmented reality, and virtual reality scenarios.

In this paper, we present OVO, an Open-Vocabulary Online mapping algorithm, which we integrate into two distinct visual SLAM pipelines. An example of our online reconstruction results is shown in Fig. 1. Our method processes RGB-D keyframes to generate 3D segments, each associated with a CLIP embedding. These segments are initialized by back-projecting masks predicted by Segment Anything Model (SAM) 2.1 [15], and are tracked over time by projecting them into 2D and matching against new masks. Each 3D segmentâs CLIP descriptor is selected from the keyframe views with the best visibility. Additionally, we introduce a novel model to extract per-instance CLIP descriptors directly from images, which are then assigned to the corresponding 3D masks. Our CLIP merging employs a neural network that learns per-dimension weighting to fuse CLIP descriptors of the same instance, while effectively generalizing to unseen classes and environments. Our pipeline not only operates online and supports loop-closure optimization, but also outperforms existing baselines in segmentation accuracy.

<!-- image-->  
â ceiling lamp â bottle â telephone â shelf â wall â chair â door â box â whiteboard â ceiling â cabinet â blinds â socket â heater â table â floor â window  
Fig. 1. OVO mapping. Given a RGB-D set of keyframes (top), our method successively reconstructs a 3D open-vocabulary representation of a scene over time (middle). At any moment, both semantic labels (bottom left) as well as instance labels (bottom right) can be effectively recovered.

## II. RELATED WORK

Our OVO estimates consistent 3D open-vocabulary semantics and seamlessly integrates with SLAM pipelines. Unlike previous methods that either use a closed set of categories, offline processing, 2D semantic representations or odometry. Tab. I provides a comparative summary of recent related works based on these aspects, with further details discussed in the remainder of this section.

Open-Vocabulary Image Semantics. The introduction of Contrastive Language-Image Pretraining (CLIP) [8], which encodes image and text tokens into a shared latent space, revolutionized semantic segmentation. By computing similarity to text inputs, CLIP enables classification into any category expressible in language. Several variations of CLIP have enhanced its performance [16, 17] and improved feature granularity, aiming to generate dense feature vectors [18, 19] rather than per-image representations. While closedvocabulary methods outperform on predefined sets, openvocabulary offers optimization-free generalization, highly relevant for diverse applications.

Offline 3D Open-Vocabulary from 3D point clouds. Most open-vocabulary 3D semantic approaches assume a known 3D point cloud. OpenScene [9] leverages OpenSeg [20] to compute CLIP features from images and trains a network to associate 2D pixels with 3D points. For each 3D point it performs average pooling on CLIP vectors from multiple views and supervises an encoder to directly assign CLIP features to 3D point clouds. OpenMask3D [10] selects k views per object, crops its 2D SAM mask to compute a CLIP features, and then features are average-pooled across crops and views. Open3DIS [11] integrates SuperPoint [21] with 2D instance segmentations and a 3D instance segmentator to generate multiple 3D instance proposals, describing each with CLIP features following OpenMask3D [10]. In contrast, OpenYolo-3D [22] uses a 2D open-vocabulary object detector instead of relying on 2D instance masks and CLIP features. It classifies each object based on the most common class across all views. While this approach eliminates the need for CLIP feature extraction, it limits each scene to a predefined set of classes.

Offline 3D Open-Vocabulary from RGB and RGB-D. OpenNeRF [23] optimizes a NeRF to encode the scene representation along with per-pixel CLIP features from OpenSeg. The OpenSeg features are projected into 3D to compute the mean and covariance of 3D points. The NeRF then renders novel views, prioritizing areas with high covariance to compute additional OpenSeg features and refine the model. Hierarchical Open-Vocabulary 3D Scene Graphs (HOV-SG) [24] relies on an offline hierarchical global fusion approach that requires precomputing 3D segments and features for all frames. These 3D segments and features are incrementally fused by merging observations across consecutive frames. The authors argue that relying solely on masked segments, as in Concept-Graphs [13], discards crucial contextual information. To address this, they propose a descriptor that merges in a handcrafted manner three CLIP embeddings per mask: (1) the full image, (2) the masked segment without background, and (3) the masked segment with background. We adopt this strategy, and contribute by proposing a novel approach to learn the CLIP merging operation.

Online Semantics. To date, online semantic methods have focused mostly on closed vocabularies. SemanticFusion [3] was one of the first semantic SLAM pipelines, predicting perpixel closed-set categories and fusing predictions from different views in 3D space. Fusion++ [25] uses Mask-RCNN [26] to initialize per-object Truncated Signed Distance Functions (TSDFs), building a persistent object-graph representation. In contrast, PanopticFusion [27] combines predicted instances and class labels (including background) to generate pixelwise panoptic predictions, which are then integrated into a 3D mesh. More recent works, such as those by Menini et al. [28] and ALSTER [29], jointly reconstruct geometry and semantics in a SLAM framework. Additionally, NIS-SLAM [30] trains a multi-resolution tetrahedron NeRF to encode color, depth and semantics. NEDS-SLAM [31] is a 3DGS-based SLAM system with embedded semantic features to learn an additional semantic representation of a closed set of classes. Similarly, Hi-SLAM [32] and SGS-SLAM [7] augment a 3DGS SLAM with semantic ids of predefined set of classes. These approaches either assume known 2D ground-truth closed set of semantic classes (and therefore only tackle a multi-view fusion problem), or only represent 2D semantics, with limited capabilities for 3D segmentation or precise 3D object localization. More recently, OpenFusion [14] and Concept-Graphs [13] integrated open-vocabulary semantic descriptors into online 3D mapping pipelines. Concept-Graphs relies on the naive maskcropping to compute CLIP descriptors, while OpenFusion uses SEEM [33] and creates a TSDF with 3D segments. None of them, however, addresses the integration into a full SLAM pipeline with loop closure optimization as we do.

TABLE I  
OVERVIEW OF 3D SEMANTIC RECONSTRUCTION BASELINES.
<table><tr><td>Method</td><td>Open Vocabulary</td><td>3D semantics</td><td>Online</td><td>Loop Closure</td></tr><tr><td>OpenScene [9]</td><td></td><td></td><td></td><td></td></tr><tr><td>OpenMask3D [10]</td><td></td><td></td><td></td><td></td></tr><tr><td>Open3DIS [11]</td><td></td><td></td><td></td><td></td></tr><tr><td>HOV-SG [24]</td><td></td><td></td><td></td><td></td></tr><tr><td>OpenNeRF [23]</td><td></td><td></td><td>xxxxx1</td><td></td></tr><tr><td>NEDS-SLAM [31]</td><td></td><td></td><td></td><td></td></tr><tr><td>NIS-SLAM [30]</td><td>X</td><td></td><td></td><td></td></tr><tr><td>SGS-SLAM [7]</td><td>X</td><td></td><td></td><td></td></tr><tr><td>Kimera-VIO [4]</td><td></td><td></td><td></td><td></td></tr><tr><td>Concept-Fusion [12]</td><td></td><td></td><td></td><td></td></tr><tr><td>Concept-Graphs [13]</td><td></td><td></td><td></td><td></td></tr><tr><td>Open-Fusion [14]</td><td></td><td></td><td></td><td></td></tr><tr><td>-oVO (ours)</td><td></td><td></td><td></td><td></td></tr></table>

## III. OVO METHODOLOGY

OVO relies on a parallel-tracking-and-mapping architecture, as first defined by Klein and Murray [34] and adopted by most visual SLAM implementations [35]. Fig. 2 shows an overview of OVO. It takes as input a stream of RGB-D keyframes $( \{ k _ { 0 } , \ldots , k _ { n } \}$ in the figure) and their respective poses and local point clouds. From this 3D representation, Sec. III-A, OVO extracts and tracks a set of 3D segments covering the whole representation (3D segment mapper in the figure, detailed in Sec. III-B). We compute a CLIP descriptor per each segmentâs viewpoint merging 3 different CLIPs (CLIP merging in the figure, detailed in Sec. III-E). Then assign to the 3D segment the most representative descriptor, Sec. III-D. When the SLAM module performs a loop closure or bundle-adjustment optimization, a routine searches for repeated 3D segments, and fuses those that were not correctly tracked, Sec. III-C.

<!-- image-->  
Fig. 2. Overview. From a stream of RGB-D keyframes, OVO builds, online, a 3D semantic representation of the scene. It relies on a 3D segment mapper to cluster 3D points into 3D segments; a queue to distribute the CLIP extraction computation, and a novel CLIP merging method to aggregate CLIP descriptors from multiple keyframes into one for each 3D segment.

## A. Map Definition

Its input is an RGB-D video $\mathcal { V } = \{ f _ { 0 } , \dotsc , f _ { \tau } \} , \ f _ { \tau } \ \in$ $\mathbb { N } _ { < 2 5 5 } ^ { w \times h \times 3 } \times \mathbb { R } _ { > 0 }$ representing the RGB-D frame of size $w \times h$ captured at time step Ï . A SLAM front-end estimates in realtime the pose $T _ { n }$ of every frame $f _ { \tau }$ in the world reference frame. The SLAM back-end selects a set of keyframes $\kappa =$ $\{ k _ { 0 } , \dots k _ { n } \} \subset \mathcal { V }$ from which it iteratively refines their poses $\mathcal { T } = \{ T _ { 0 } , \ldots , T _ { n } \} , T _ { n } \in S E ( 3 )$ asynchronously, at a rate lower than the video rate of the tracking thread.

Our scene representation or âmapâ $\mathcal { M } ~ = ~ \{ \mathcal { T } , \mathcal { P } , \mathcal { S } \}$ , consists on these keyframe poses $\tau$ , a point cloud $\mathcal { P } =$ $\{ P _ { 0 } , \ldots , P _ { m } \}$ and a set of 3D segments $\mathcal { S } = \{ S _ { 0 } , \ldots , S _ { q } \}$ , being q the identifier of the last added segment. Every map point $\begin{array} { r } { P = \left( \left[ x \underline { {  { y } } } z \right] ^ { \top } , l _ { p } \right) } \end{array}$ is defined by its 3D coordinates $\left[ \begin{array} { l l l } { x } & { y } & { z } \end{array} \right] ^ { \mathrm { ~ ! ~ } } \in \mathbb { R } ^ { 3 }$ and a discrete label $l _ { p } \in$ $\{ - 1 , 0 , 1 , \ldots , \bar { q } \} , \ l _ { p } \ \bar { > } \ - 1$ indicating the 3D segment the point belongs to, and $l _ { p } = - 1$ indicating that it is unassigned. The dense point cloud P is built concatenating at each keyframe $k _ { n }$ the estimated 3D points $\mathcal { P } _ { n }$ P provided by the SLAM front-end. If the SLAM front-end does not estimate a dense point cloud, $\mathcal { P } _ { n }$ is computed as the unprojection of the input depth map to 3D using the estimated camera pose $T _ { n } \in S E ( 3 )$ . To avoid P growing unconstrained, a pixel is not projected to 3D if a previously unoccluded 3D point falls inside its neighborhood when projected back to 2D. For every 3D point, occlusion is assessed by comparing its projected depth to its measured depth in the 2D pixel it is projected. Every 3D segment $S = \left( \mathbf { d } , \kappa \right)$ has a unique identifier, its semantics are described by a CLIP feature $\textbf { d } \in \mathbb { R } ^ { d }$ , and stores in a heap Îº the indices of the best keyframes in which $S$ was seen, ordered by visibility scores.

## B. 3D Segment Mapper

For every new keyframe $k _ { n } ,$ we run an image segmentation model that returns a set of 2D segments $\begin{array} { l l } { { \tilde { \cal S } } _ { n } } & { = } \end{array}$ $\{ ( s _ { 0 } , l _ { s 0 } ) , ( s _ { 1 } , l _ { s 1 } ) , . . . \}$ , each segment being composed of a mask s and a label $l _ { s } .$ , which is initialized as $l _ { s } : = - 1$

Algorithm 1 3D Segment Mapper   
1: function 3D SEGMENT MAPPER $( { \mathcal { P } } , S , k _ { n } , T _ { n } )$   
2: SËn â segment keyframe(kn)   
3: $\tilde { \mathcal { P } } _ { n } \gets$ project point cloud $\mathsf { l } ( \mathcal { P } , T _ { n } )$   
4: for $( s , l _ { s } )$ in $\dot { \tilde { S _ { n } } }$ do â· For every 2D segment in $k _ { n }$   
5: mode, v â get label mode and votes $( \tilde { \mathcal { P } } _ { n } , s , \epsilon )$   
6: if v > Ïµ then â· #votes greater than threshold   
7: if mode = â1 then   
8: Sq+1 â new 3D segment $( q + 1 , n , s )$   
9: $S ^ { ' } { \gets } S \cup \{ S _ { q + 1 } \}$   
10: $l _ { s } \gets q + 1$   
11: else   
12: S â update 3D segment(Smode, n, s)   
13: $l _ { s } \gets z _ { l }$   
14: ${ \tilde { S } } _ { n } \gets$ merge and prune 2D segments $( { \tilde { \cal S } } _ { n } )$   
15: P â update pcd label $\mathsf { \Omega } _ { \mathsf { i } } ( \mathcal { P } , \tilde { \mathcal { P } } _ { n } , \tilde { S } _ { n } )$   
16: return $\bar { \mathcal { P } } , \mathcal { S } , \tilde { \mathcal { S } } _ { n }$

We then select the 3D map points in $k _ { n } \mathrm { ' s }$ frustum, project them to $k _ { n } ,$ and remove occluded points by comparing their projected depth to the input depth. In this manner, we obtain the 2D point set $\tilde { \mathcal { P } } _ { n } = \{ p _ { 0 } , p _ { 1 } , . . . \}$ , for which $\boldsymbol { p } ~ = ~ \left( \begin{array} { l l } { \left[ u } & { v \right] ^ { \top } , } &  l _ { p } \right) \end{array}$ . We compute the label mode of all points p within a segment s, that we will represent slightly abusing notation as $z _ { l } : = \arg \operatorname* { m a x } _ { l _ { p } } \left( \tilde { \mathcal { P } } \cap s \right)$ . If the mode receives less votes v than a predefined threshold Ïµ, we discard s. If not, two possibilities can occur:

1) If $z _ { l } = - 1$ , we set $z _ { l } : = q + 1$ and initialize a new 3D segment $S _ { q + 1 }$ with an empty CLIP feature d (filled later as described in Sec. III-D), and a keyframe heap $\kappa : = \{ ( n , r ) \}$ , initialized with $k _ { n } \mathrm { ' s }$ index and $s ^ { \prime }$ visibility score r.

2) Otherwise, 2D segment s is a match for 3D segment $S _ { z _ { l } }$ and the keyframe will be inserted into $\kappa ,$ and stored if it is one of the best views or if Îº is not full.

For both, the unassigned 3D points and 2D segmentâs labels, $l _ { p }$ and $l _ { s }$ are updated to the identifier of the matched $S _ { z _ { l } }$ . After matching all 2D masks, those that share the same $l _ { s }$ are merged. Finally, once all masks are gathered in ${ \tilde { S } } _ { n }$ the tuple $\bar { ( } k _ { n } , \tilde { S } _ { n } )$ is pushed to the queue Q. Keyframes and masks remain in Q until processing resources become available to compute the CLIP descriptors for the highest-

<!-- image-->  
Fig. 3. Out-of-distribution queries. From left to right, top to bottom, observe how common-language queries allow to differentiate bins based on a recycling symbol; recongize sofas and chairs as places to sit; that you can take a nap in a sofa, pillows and couches are soft objects, and books are readable, that the clock tells the hour, the blackboard is to draw equations, and the jacket is something to stay warm. Colorbar shows similarity strength.

scoring 2D segments.

## C. Loop Closure

When the SLAM module closes a loop or completes a Global Bundle Adjustment, OVO updates both its map and the set of 3D instances. We denote both after the update as $\mathcal { M } ^ { \prime }$ and $S ^ { \prime }$ . For each updated keyframe $T _ { n } ^ { \prime } ~ \in ~ \mathcal { T } ^ { \prime }$ , its associated local point cloud is also updated by propagating the pose correction as $\mathcal { P } _ { n } : = T _ { n } ^ { \prime } ~ T _ { n } ^ { - 1 } ~ \mathcal { P } _ { n }$ . This transforms the points from the world frame to the original keyframeâs and back using the updated pose $T _ { n } ^ { \prime }$ . Keyframes that are removed during SLAM optimization are discarded along with their associated 3D points. After updating the 3D points, the temporary queue Q is cleared. Next, the set of 3D instances $S ^ { \prime }$ is pruned by removing instances whose associated points were entirely deleted during optimization. Following, instance fusion is performed by comparing remaining pairs of 3D instances. Two instances are merged if they satisfy the following criteria: (1) The distance between their point cloud centroids is < 150cm, (2) the cosine similarity between their CLIPs is $> 0 . 8 .$ , and (3) more than 50% of their points lie within 10cm of a point in the other instance. For a pair of segments $S _ { i }$ and $S _ { j }$ to be merged, their point indices are unified as $\begin{array} { r } { \kappa _ { i } : = \kappa _ { i } \cup \kappa _ { j } . } \end{array}$ , and all map points previously labeled as j are reassigned to i, i.e., $\forall P _ { k } \in \mathcal { P } | l _ { k } = j , \implies l _ { k } : = i .$

## D. CLIP Descriptors

When a tuple $\left( k _ { q } , \tilde { S } _ { q } \right)$ is popped from Q, only the matched 2D segments for which $k _ { q }$ is still in the Îº of their 3D instance S are selected. A CLIP descriptor d is computed for each of them as explained in Sec. III-E. Then, the final descriptor for a 3D segment S is selected between the 2D segments in its keyframesâ heap $\kappa ,$ as the CLIP descriptor with the smallest aggregated distance to the rest. To query the 3D semantic representation, text queries are encoded to CLIP space. Then, we compute the cosine similarity between the CLIP descriptor of the query and the descriptor d of each 3D segment in S.

## E. CLIP Merging

Similarly to HOV-SG [24], for each 2D segment we compute three CLIP descriptors: 1) ${ \bf d } _ { 0 }$ for the full keyframe, 2) ${ \bf d } _ { 1 }$ for the segment masking the rest of the image out, and 3) ${ \bf d } _ { 2 }$ for the minimum bounding box that contains the segment. In contrast, in our case, the CLIP descriptor d = $\textstyle \sum _ { i = 0 } ^ { \cdot 2 } { \mathbf { w } } _ { i } \odot \mathbf { d } _ { i }$ of a 2D segment is the result of merging the three descriptors $\mathbf { d } _ { i = \{ 0 , 1 , 2 \} }$ using a per-dimension weighted average with weights $\mathbf { \bar { w } } _ { i } \in \mathbb { R } ^ { d }$ (â is the Hadamard product). Our weights $\mathbf { w } _ { i = \{ 0 , 1 , 2 \} }$ are predicted by a neural model, as shown in Fig. 2. Note that HOV-SGâs merging is done with hand-crafted scalar weights $\begin{array} { r } { ( i . e . , \mathbf { d } = \sum _ { i = 0 } ^ { 2 } { w _ { i } \mathbf { d } _ { i } } , w _ { i } \in \mathbb { R } ) } \end{array}$

As seen in Fig. 2, the input to our CLIP merging is three CLIPs $\mathbf { d } _ { i = \{ 0 , 1 , 2 \} }$ . These are first passed by a transformer encoder, and the output is flattened and fed to a MLP, predicting the weights, and a softmax, forcing $\begin{array} { r } { \sum _ { i = 0 } ^ { 2 } { \bf w } _ { i } = } \end{array}$ $\mathbf { 1 } ^ { d } .$ . Our CLIP merging is pre-trained following SigLIP [16]. For a mini-batch $B ~ = ~ \{ ( s _ { 0 } , c _ { 0 } ) , ( s _ { 1 } , c _ { 1 } ) , \ldots \}$ composed by pairs of 2D segments $s _ { j }$ and semantic classes $c _ { j }$ , we minimize the sigmoid cosine similarity loss

$$
L = - \frac { 1 } { | \mathcal { B } | } \sum _ { i = 1 } ^ { | \mathcal { B } | } \sum _ { j = 1 } ^ { | \mathcal { B } | } \log \left( \frac { 1 } { 1 + \exp ( z _ { i j } ( - t \mathbf { d } _ { i } \cdot \mathbf { y } _ { j } + b ) ) } \right)\tag{1}
$$

between the merged CLIP descriptor $\mathbf { d } _ { i }$ , and the CLIP embedding $\mathbf { y } _ { j }$ of the semantic class $c _ { j }$ associated to the 2D segment $s _ { j }$ in the same batch B. $z _ { i j }$ is the label for a given image and class input, which equals 1 if they are paired and â1 otherwise. b and t are learnable bias and temperature parameters, used to compensate the imbalance coming from negative pairs dominating the loss.

## IV. EXPERIMENTS

First, we report OVO evaluation on 3D online semantic mapping on two established datasets, one synthetic (Replica), and one real (ScanNetv2). Then, we present our CLIP merging evaluation on semantic classification of images with ground-truth segmentation masks both on a dataset with multiple masks per image (ScanNet++) and with a single mask per image (ImageNet-S), and against alternative methods integrated into OVO for 3D semantic mapping (Replica).

<!-- image-->  
Fig. 4. 3D semantic segmentation on Replica. OVO yields more accurate results in comparison to the two best offline baselines.

Implementation. For OVO, we implemented three different configurations to show its flexibility: (1) OVO-mapping, that uses ground-truth camera poses, (2) OVO-Gaussian-SLAM, where we integrate our contributions within Gaussian-SLAM [36], a SLAM method targeting novel-view synthesis and dense point cloud reconstruction, although not realtime, and (3) OVO-ORB-SLAM2 for which we integrate with ORB-SLAM2 [37], a real-time feature-based SLAM system with loop-closure. While OVO-Gaussian-SLAM uses the center of 3D Gaussians as the dense point cloud P, for OVO-ORB-SLAM2 we build a dense point cloud by registering the local point clouds from the RGB-D images. All three configurations use SAM2.1-l for 2D segmentation and SigLip ViT-SO400 for CLIP descriptors. Our CLIP merging has 5 self-attention layers with 8 heads, a 1152 latent dimension, with drop-out of 0.1, and 4 layers MLP with 3Ã1152 input/output neurons and Ã4 inverse bottleneck with Leaky ReLU activations. It was trained with 4 Nvidia V100 GPUs, using Pytorch, with AdamW optimizer, learning rate 1 Ã 10â6, gradient clipping at 1, and batch size of 512 per GPU, for 15 epochs using the top 100 semantic labels from ScanNet++ 250 training set. To compensate for class imbalance, in the loss we weight each element of the batch by the inverse of their class frequency in the training set.

Baselines. We evaluate CLIP merging against baselines that compute local CLIP descriptors [12, 13, 24] (using all of them SigLIP-SO400M) and Alpha-CLIP [19], a state-ofthe-art model developed to condition CLIP using masks. Additionally, we include two variations of our CLIP merging trained in the same setup, in order to validate its design: directly predicting the fused descriptor, and predicting only one weight per descriptor. As detailed in Sec. II, existing semantic SLAM pipelines do not construct a 3D representation that can be evaluated using 3D metrics for open-set classes. Thus, we compare OVO against similar 3D openvocabulary online mapping systems, Concept-Graphs [13], and OpenFusion [14]; and the state-of-the-art 3D openvocabulary offline baselines OpenScene [9], OpenNeRF [23], Open3DIS [11] and HOV-SG [24]. Finally, we evaluate computational cost against Concept-Graphs, OpenFusion, HOV-SG and OpenNeRF, but exclude Open3DIS and OpenScene, as they rely on pre-processed 3D geometry and features.

Datasets. ScanNet++ [38] has 250 training and 50 validation indoor RGB+D scenes sequences. We use 2D rasterized masks for a total of 1.6M and 400k 2D instance samples respectively. Semantic classes are mapped into either the set of 100 most commons (used for training) or the full set of over 1.6k classes (used for evaluation). ImageNet-S [39] has a validation set of â¼ 12k images with 919 semantic labels. ScanNetv2 [40] has a full validation set of 312 RGB+D sequences of real scenes (FVS). We also evaluate on the 5-scene subset used by HOV-SG (HVS). We use the original annotation set with 20 classes (ScanNet20) and the expanded set with 200 classes (ScanNet200 [41]). On Replica [42], we use the standard 8-scene subset (office-0...4, room-0...2) and its 51 annotated classes.

Metrics. Semantic classification is evaluated using mean Intersection Over Union (mIoU) and mean Accuracy (mAcc) on ScanNet++, while on ImageNet-S we report the standard Top-1 and Top-5 mAcc. While we assess CLIP merging in 2D to isolate other factors, the full OVO is evaluated in 3D by labeling the vertices of ground-truth meshes and comparing them against ground-truth 3D labels. For Replica, following OpenNeRF [23], we report mIoU and mAcc, categorizing labels into tertiles based on their frequency (head, common, and tail). In ScanNetv2, we further present metrics weighted by the label frequency in the ground truth (f-mIoU and f-mAcc). Additionally, we analyze our computational footprint. We measure wall-clock time required to optimize Replica scenes, as well as mean and max GPU vRAM and max system RAM usage (in GB). Each table highlights first , second , and third best results.

TABLE II  
3D SEMANTIC SEGMENTATION EVALUATION ON REPLICA 51 CLASSES, SPLITTING BY FREQUENCY TERTILES: HEAD, COMMON AND TAIL.
<table><tr><td rowspan="2"></td><td rowspan="2">Online</td><td rowspan="2">Geo- metry</td><td rowspan="2">Camera pose / ATE RMSE [cm]</td><td colspan="2">All</td><td colspan="2">Head</td><td colspan="2">Common</td><td colspan="2">Tail</td></tr><tr><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td></tr><tr><td>OpenScene[ [9] (Distilled)</td><td>X</td><td>GT</td><td>GT</td><td>14.8</td><td>23.0</td><td>30.2</td><td>41.1</td><td>12.8</td><td>21.3</td><td>1.4</td><td>6.7</td></tr><tr><td>OpenScene [9] (Ensemble)</td><td>X</td><td>GT</td><td>GT</td><td>15.9</td><td>24.6</td><td>31.7</td><td>44.8</td><td>14.5</td><td>22.6</td><td>1.5</td><td>6.3</td></tr><tr><td>penNeR [23]</td><td>X</td><td>Est.</td><td>GT</td><td>20.4</td><td>31.7</td><td>35.4</td><td>46.2</td><td>20.1</td><td>31.3</td><td>5.8</td><td>17.6</td></tr><tr><td>HOV-SG [24]</td><td>X</td><td>Est.</td><td>GT</td><td>22.5</td><td>34.2</td><td>35.9</td><td>44.2</td><td>23.6</td><td>42.3</td><td>8.0</td><td>16.1</td></tr><tr><td>Open3DIS [11] (SigLip</td><td>X</td><td>GT</td><td>GT</td><td>25.6</td><td>38.7</td><td>49.7</td><td>64.4</td><td>_22.1</td><td>_42.4</td><td>4.9</td><td>9.4</td></tr><tr><td>Concept-Graphs [13</td><td></td><td>Est..</td><td>GT</td><td>16.7</td><td>33.7</td><td>27.3</td><td>39.1</td><td>15.1</td><td>35.4</td><td>4.4</td><td>26.8</td></tr><tr><td>Open-Fusion [14]</td><td></td><td>Est.</td><td>GT</td><td>20.5</td><td>34.8</td><td>37.9</td><td>51.7</td><td>14.0</td><td>30.3</td><td>9.8</td><td>22.2</td></tr><tr><td>OVO-mapping (ours)</td><td></td><td>Est.</td><td>GT</td><td>27.0</td><td>39.1</td><td>45.0</td><td>59.9</td><td>25.1</td><td>_38.5</td><td>11.0</td><td>18.8</td></tr><tr><td>ovO-Gaussian-SLAM (ours)</td><td></td><td>Est. </td><td>0.6</td><td>27.1</td><td>38.6</td><td>44.1</td><td>58.0</td><td>25.0</td><td>39.0</td><td>12.1</td><td>188.9</td></tr><tr><td>OVO-ORB-SLAM2 (ours)</td><td></td><td>Est.</td><td>1.9</td><td>25.6</td><td>39.0</td><td>43.0</td><td>59.1</td><td>21.6</td><td>38.3</td><td>12.1</td><td>19.6</td></tr></table>

TABLE III

3D SEMANTIC SEGMENTATION ON SCANNETV2 WITH FREQUENCY WEIGHTED METRICS ON 5 (HVS) AND ALL 312 VAL. SCENES (FVS).
<table><tr><td rowspan="2">Method</td><td rowspan="2">Online</td><td rowspan="2">Geo- metry</td><td rowspan="2">Camera pose / ATE RMSE [cm]</td><td colspan="4">ScanNet20</td><td colspan="4">ScanNet200</td></tr><tr><td>mIoU</td><td>mAcc</td><td>f-mIoU</td><td>f-mAcc</td><td>mIoU</td><td>mAcc</td><td>f-mIoU</td><td>f-mAcc</td></tr><tr><td rowspan="9">Open3DIS [11] (SigLip) OpenScene(Ensemble) [9] HOV-SG [24]</td><td>X X</td><td>GT</td><td>GT</td><td>37.3</td><td>52.8</td><td>57.0</td><td>67.9</td><td>17.8</td><td>23.7</td><td>27.9</td><td>34.1</td></tr><tr><td></td><td>GT</td><td>GT</td><td>44.6</td><td>61.9</td><td>57.6</td><td>71.0</td><td>9.4</td><td>12.6</td><td>27.8</td><td>32.0</td></tr><tr><td></td><td>Est.</td><td>GT GT</td><td>34.4 17.1</td><td>51.1</td><td>47.3</td><td>61.8</td><td>11.2</td><td>18.7 11.7</td><td>27.7</td><td>37.6</td></tr><tr><td>Concept-Graphs [13]</td><td></td><td>Est.</td><td></td><td>29.1</td><td>26.0</td><td>33.1</td><td></td><td>6.0</td><td>21.4</td><td>27.7</td></tr><tr><td>Open-Fusion [14]</td><td></td><td>Est.</td><td></td><td>30.1 39.9</td><td>54.1</td><td>68.1</td><td></td><td>8.6 12.8</td><td>38.4</td><td>47.9</td></tr><tr><td>ovO-mapping (ours)</td><td></td><td>GT Est. GT</td><td>38.1</td><td>50.5</td><td>57.6</td><td>70.5</td><td>17.2</td><td>25.3</td><td>45.4</td><td>56.4</td></tr><tr><td>OVO-Gaussian-SLAM (ours)</td><td></td><td>Est.</td><td>23.7</td><td>29.3</td><td>41.1 43.0</td><td>59.5</td><td></td><td>11.8 18.8</td><td>30.1</td><td>42.6</td></tr><tr><td>OVO-ORB-SLAM2 (ours)</td><td></td><td>Est.</td><td>21.5</td><td>31.3</td><td>45.2 45.8</td><td>61.2</td><td></td><td>13.6 22.2</td><td>38.2</td><td>51.0</td></tr><tr><td>OVO-ORB-SLAM2 w/o loop clos.</td><td>â</td><td>Est.</td><td>30.2</td><td>23.6</td><td>34.5 41.4</td><td>56.9</td><td></td><td>10.3 17.3</td><td>33.2</td><td>46.0</td></tr><tr><td rowspan="3">Open3DIS [11] (SigLip) PAS</td><td></td><td>GT GT</td><td>GT</td><td>24.7</td><td>40.9</td><td>32.5</td><td>45.3</td><td>9.4</td><td>17.0</td><td>22.9</td><td>32.2</td></tr><tr><td>OpenScene(Ensemble) [9]</td><td>X</td><td>GT</td><td>47.0</td><td>70.3</td><td>57.7</td><td>69.8</td><td>11.6</td><td>22.8</td><td>24.5</td><td>_29.2</td></tr><tr><td>OVO-mapping (ours)</td><td>Est.</td><td>GT</td><td></td><td>37.3 58.9</td><td>55.13</td><td>69.4</td><td></td><td>17.4 35.9</td><td>44.3</td><td>57.8</td></tr></table>

## A. 3D Semantic Segmentation

Replica. Tab. II presents segmentation results for all our OVO configurations alongside relevant baselines. OVO outperforms all baselines in the aggregated mIoU and mAcc (âAllâ column). OVO-Gaussian-SLAM and OVO-ORB-SLAM2 surpass both offline and online mapping algorithms. This is particularly noteworthy since both implementations estimate camera poses and scene geometry, whereas all baselines (indicated in the table) rely either on groundtruth geometry, camera pose, or both. Thanks to the strong generalization of our CLIP merging, all OVO implementations have a significantly better mIoU on tail categories, which demonstrates less false-positives. As shown in Fig. 4, OVO effectively segments and classifies 3D instances, such as chairs and tables, that other baselines often misclassify due to the excessive context information incorporated into CLIP descriptors. OVO even outperforms the ground truth in some instances. For example, in âoffice4â (top left of Fig. 4), the ground-truth label for the table is missing, and one chair is misclassified as the floor. This underscores the advantage of open-set pipelines, particularly in situations where previous SLAM algorithms, which rely on known 2D semantics [7, 30], would fail.

ScanNetv2. Results, summarized in Tab. III, show how OVO-mapping matches HOV-SG, and even Open3DIS in the set ScanNet20. On the harder set ScanNet200, OVOmapping has a similar performance to Open3DIS in mIoU, although it is significantly better in terms of f-mIoU and f-mAcc. OpenScene does achieve the best performance on ScanNet20. Nevertheless, its significant drop when using the extended set of classes highlights a weaker generalization capabilities than OVO and other baselines.

SLAM comparison. The difference between OVOâs two SLAM versions and OVO-mapping is bigger in ScanNetv2 than in Replica (compare Tab. II and Tab. III), due to image blur and noisy depths in ScanNetv2. Gaussian-SLAM benefits from a more complex strategy for densification and pruning of the 3D point cloud, outperforming our simpler depth unprojection in Replica. However, while its camera tracking works flawlessly there, it does struggle in ScanNetv2 noisier images, where loop-closure plays a key role. Comparing OVO-ORB-SLAM2 w/o and w/ loop-closure, Tab. III, shows the importance of this feature. Further, Fig. 5 illustrates the loop closure correction over inconsistent reconstructions with repeated semantic instances, caused by odometric drift.

Computational footprint. Despite OpenFusion being 2Ã faster than OVO, thanks to using SEEM instead of SAM+SigLIP, OVO achieves a better balance between speed and performance. It is still 2.5Ã faster than Concept-Graphs, 3Ã faster than OpenNerf and 80Ã faster than HOV-SG, as shown in Tab. IV. In contrast with HOV-SG, that relies on an expensive hierarchical merging of segments, requiring almost Ã10 more RAM, OVO shows a lower RAM and GPU vRAM usage that enables its use on consumer devices. OVO-ORB-SLAM2 takes on average 0.67 seconds per keyframe on Replica and ScanNetv2, spiking up to 1.4 seconds for the slowest frame, and up to up to 2.5 and 6.1 seconds after loop closure on âscene0011 00â and âscene0231 00â respectively. This is compatible in our experiments with a conservative keyframe creation policy of 1 keyframe every 10 frames. Therefore, it is compatible with real-time SLAM pipelines, in which the critical camera tracking runs at video rate while the mapping runs at lower frequencies.

Before loop-closure  
<!-- image-->

<!-- image-->  
Fig. 5. Visualization of OVO-ORB-SLAM2 loop closure on âscene0011 00â (ScanNet). We highlight four instances split due to tracking drift and effectively merged after loop-closure by our semantic fusion.

TABLE IV  
RUNTIME STATISTICS ON REPLICA WITH 2K FRAMES PER SCENE.
<table><tr><td>Method</td><td>vRAM Avg / Max</td><td>RAM Max</td><td>Time Avg</td></tr><tr><td>HOV-SG [24]</td><td>6 / 12 GB</td><td>139 GB</td><td>~11h</td></tr><tr><td>OpenNeRF [23]</td><td>4 / 22 GB</td><td>44 GB</td><td>~20m</td></tr><tr><td>Open-Fusion [14]</td><td>37 4 GB</td><td>6 GB</td><td>~3m</td></tr><tr><td>CG [13]</td><td>7 / 11 GB</td><td>16 GB</td><td>â¼16m</td></tr><tr><td>OVO-mapping (ours)</td><td>4/ 8 GB</td><td>12 GB</td><td>~6m</td></tr></table>

## B. CLIP Merging

In Tab. V, we report evaluation on ImageNet-S, including also Alpha-CLIP, and on which both HOV-SGâs and Concept-Fusionâs merging approaches are equivalent to just computing the global descriptor due to there being only one mask per image. Tab. VI presents 2D semantic classification results on unseen scenes from ScanNet++, using the expanded label set of 1.6k, of our CLIP merging vs. HOV-SGâs and Concept Fusionâs (CF) CLIP merging, and the simpler mask crop used by Concept-Graphs.

Overall, ours outperforms baselines, particularly in frequency-weighted metrics, although with slightly worse mAcc in ScanNet++. Alpha-CLIP performs worse than simpler approaches like our CLIP merging. Using a better backbone (SigLIP-SO400M vs ViT-L/14) outperforms a significantly more expensive fine-tuning (our trained two day on 4 V100 vs their on 128 A100 GPUs).

TABLE V  
IMAGENET-S SEMANTIC CLASSIFICATION ACCURACY.
<table><tr><td>Method</td><td>Top-1 mAcc</td><td>Top-5 mAcc</td></tr><tr><td>Alpha-Clip (ViT-L/14@336)[19]</td><td>77.6</td><td>94.1</td></tr><tr><td>SigLIP-SO400M[16]</td><td>82.5</td><td>95.7</td></tr><tr><td>Mask crop</td><td>80.3</td><td>93.4</td></tr><tr><td>Our CLIP merging</td><td>84.8</td><td>96.6</td></tr></table>

TABLE VI

2D OPEN VOCABULARY SEMANTIC CLASSIFICATION ON SCANNET++.
<table><tr><td>Method</td><td>mIoU</td><td>mAcc</td><td>f-mIoU</td><td>f-mAcc</td></tr><tr><td>Concept-Fusion[12]</td><td>8.6</td><td>17.0</td><td>10.0</td><td>12.8</td></tr><tr><td>Mask crop</td><td>8.5</td><td>17.0</td><td>10.0</td><td>12.8</td></tr><tr><td>HOV-SG merging[24]</td><td>9.4</td><td>15.9</td><td>12.8</td><td>15.9</td></tr><tr><td>Our CLIP merging</td><td>9.5</td><td>14.3</td><td>36.9</td><td>49.4</td></tr><tr><td>CLIP merging variations</td><td></td><td></td><td></td><td></td></tr><tr><td>- fused</td><td>6.2</td><td>11.6</td><td>40.0</td><td>56.7</td></tr><tr><td>- per-descriptor</td><td>9.0</td><td>15.6</td><td>12.7</td><td>15.9</td></tr></table>

TABLE VII

3D OPEN-VOCABULARY SEMANTIC METRICS ON REPLICA OF OVO-MAPPING WITH ALTERNATIVE CLIP MERGING.
<table><tr><td rowspan="3"></td><td colspan="2">All</td><td colspan="2">Seen</td><td colspan="2">Unseen</td></tr><tr><td></td><td>mIoU mAcc</td><td>mIoU mAcc</td><td></td><td></td><td>mIoU mAcc</td></tr><tr><td>w/ HOV-SG&#x27;s fusion</td><td>20.3</td><td>38.1</td><td>22.3</td><td>45.1</td><td>18.3</td><td>30.9</td></tr><tr><td>w/ our CLIP merging</td><td>27.0</td><td>39.1</td><td>36.7 54.7</td><td></td><td>16.9</td><td>22.8</td></tr><tr><td colspan="7">w/ CLIP merging variations</td></tr><tr><td>- fused</td><td>20.2</td><td>33.2</td><td>32.2 52.5</td><td></td><td>0.8</td><td>1.2</td></tr><tr><td>- per-descriptor</td><td>20.2</td><td>38.2</td><td>26.6</td><td>48.6</td><td>13.6</td><td>27.4</td></tr></table>

Regarding alternatives, the per-descriptor weights predictor achieves a similar performance to HOV-SG, while directly predicting a fused descriptor achieves better frequencyweighted metrics but significantly worse overall ones, which indicates overfitting. This is further validated evaluating OVO-mapping in Replica, Tab. VII using HOV-SGâs merging approach, and the alternatives to our CLIP merging. The fused predictor performance collapses in classes not seen during training, while our proposed CLIP merging has a slightly worse performance than HOV-SGâs, while being significantly better on the known classes. The per-CLIP weights is unable to match the performance, highlighting the impact of per-dimension weights.

Finally, we highlight in Fig. 3 how our CLIP merging preserves their rich semantic encoding, allowing our merged CLIPs to generalize to zero-shot complex language queries. For instance, our descriptors distinguish between two trash bins based on a recycling symbol on one of them, despite both being labeled just as bin in the ground truth.

## C. Limitations

Despite OVO state-of-the-art results on 3D indoor semantic segmentation, generalization to outdoor large-scale scenes may face challenges such as different class distributions, illumination and blur, and higher tracking errors. Our semantic fusion at loop closure effectively corrects odometric drift. However, it sometimes misses instances that should be fused, something that may be fixed by a richer and more accurately localized set of features. We also observed in CLIP merging a slight bias towards classes seen at training, which may be solved with larger training sets.

## V. CONCLUSIONS

In this paper, we present OVO, an open-vocabulary, online 3D mapping method. Our pipeline extracts 3D segments from 2D masks and tracks them across keyframes. To assign CLIP descriptors to 3D segments, we introduce a novel strategy: each 2D segment receives a single descriptor computed as a weighted sum of embeddings from the full image, the masked region, and its surrounding bounding box. The weights are predicted by a neural network, which outperforms handcrafted heuristics while retaining strong generalization. We also develop a mechanism to fuse instances that are affected by odometric drift after the geometric corrections of a loop closure. OVO outperforms existing baselines in both computational efficiency and segmentation quality across multiple datasets. By bridging SLAM with open-vocabulary representations, we believe that our work broadens the scope of applications in these two domains.

## REFERENCES

[1] S. Y. Bao et al., âSemantic structure from motion,â CVPR, 2011.

[2] A. Kundu et al., âPanoptic neural fields: A semantic objectaware neural scene representation,â CVPR, 2022.

[3] J. McCormac et al., âSemanticfusion: Dense 3d semantic mapping with convolutional neural networks,â ICRA, 2017.

[4] A. Rosinol et al., âKimera: an open-source library for realtime metric-semantic localization and mapping,â ICRA, 2020.

[5] J. Civera et al., âTowards semantic SLAM using a monocular camera,â IROS, 2011.

[6] S. Zhu et al., âSni-slam: Semantic neural implicit slam,â CVPR, 2024.

[7] M. Li et al., âSgs-slam: Semantic gaussian splatting for neural dense slam,â ECCV, 2024.

[8] A. Radford et al., âLearning transferable visual models from natural language supervision,â ICML, 2021.

[9] S. Peng et al., âOpenscene: 3d scene understanding with open vocabularies,â CVPR, 2023.

[10] A. Takmaz et al., âOpenMask3D: Open-Vocabulary 3D Instance Segmentation,â NeurIPS, 2023.

[11] P. Nguyen et al., âOpen3dis: Open-vocabulary 3d instance segmentation with 2d mask guidance,â CVPR, 2024.

[12] K. Jatavallabhula et al., âConceptfusion: Open-set multimodal 3d mapping,â RSS, 2023.

[13] Q. Gu et al., âConceptgraphs: Open-vocabulary 3d scene graphs for perception and planning,â ICRA, 2024.

[14] K. Yamazaki et al., âOpen-fusion: Real-time open-vocabulary 3d mapping and queryable scene representation,â ICRA, 2024.

[15] N. R. et al., âSam 2: Segment anything in images and videos,â arXiv:2408.00714, 2024.

[16] X. Zhai et al., âSigmoid loss for language image pre-training,â ICCV, 2023.

[17] M. Cherti et al., âReproducible scaling laws for contrastive language-image learning,â CVPR, 2023.

[18] C. Zhou et al., âExtract free dense labels from clip,â ECCV, 2022.

[19] Z. Sun et al., âAlpha-clip: A clip model focusing on wherever you want,â CVPR, 2024.

[20] G. Ghiasi et al., âScaling open-vocabulary image segmentation with image-level labels,â ECCV, 2022.

[21] D. DeTone et al., âSuperPoint: Self-Supervised Interest Point Detection and Description,â CVPRW, 2018.

[22] M. E. A. Boudjoghra et al., âOpen-YOLO 3D: Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation,â ICLR, 2025.

[23] F. Engelmann et al., âOpenNeRF: Open Set 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered Novel Views,â ICLR, 2024.

[24] A. Werby et al., âHierarchical open-vocabulary 3d scene graphs for language-grounded robot navigation,â RSS, 2024.

[25] J. McCormac et al., âFusion++: Volumetric object-level slam,â 3DV, 2018.

[26] K. He et al., âMask R-CNN,â ICCV, 2017.

[27] G. Narita et al., âPanopticfusion: Online volumetric semantic mapping at the level of stuff and things,â IROS, 2019.

[28] D. Menini et al., âA real-time online learning framework for joint 3d reconstruction and semantic segmentation of indoor scenes,â RAL, 2021.

[29] S. Weder et al., âAlster: A local spatio-temporal expert for online 3d semantic reconstruction,â WACV, 2025.

[30] H. Zhai et al., âNis-slam: Neural implicit semantic rgb-d slam for 3d consistent scene understanding,â TVCG, 2024.

[31] Y. Ji et al., âNeds-slam: A neural explicit dense semantic slam framework using 3d gaussian splatting,â RAL, 2024.

[32] B. Li et al., âHier-slam: Scaling-up semantics in slam with a hierarchically categorical gaussian splatting,â ICRA, 2025.

[33] X. Zou et al., âSegment everything everywhere all at once,â NeurIPS, 2023.

[34] G. Klein et al., âParallel tracking and mapping for small ar workspaces,â ISMAR, 2007.

[35] C. Campos et al., âORB-SLAM3: An accurate open-source library for visual, visualâinertial, and multimap slam,â TRO, 2021.

[36] V. Yugay et al., âGaussian-slam: Photo-realistic dense slam with gaussian splatting,â arXiv:2312.10070, 2023.

[37] R. Mur-Artal et al., âOrb-slam2: An open-source slam system for monocular, stereo, and rgb-d cameras,â TRO, 2017.

[38] C. Yeshwanth et al., âScannet++: A high-fidelity dataset of 3d indoor scenes,â ICCV, 2023.

[39] S. Gao et al., âLarge-scale unsupervised semantic segmentation,â TPAMI, 2022.

[40] A. Dai et al., âScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes,â CVPR, 2017.

[41] D. Rozenberszki et al., âLanguage-grounded indoor 3d semantic segmentation in the wild,â ECCV, 2022.

[42] J. S. et al., âThe Replica dataset: A digital replica of indoor spaces,â arXiv:1906.05797, 2019.

[43] Z. Zhu et al., âNice-slam: Neural implicit scalable encoding for slam,â CVPR, 2022.

[44] A. Kirillov et al., âSegment anything,â arXiv:2304.02643, 2023.

[45] M. Qin et al., âLangsplat: 3d language gaussian splatting,â CVPR, 2024.

[46] X. Zhao et al., âFast segment anything,â arXiv:2306.12156, 2023.

[47] Z. Zhang et al., âEfficientvit-sam: Accelerated segment anything model without performance loss,â CVPR, 2024.

[48] G. Ilharco et al., âOpenclip,â 2021.

[49] C. Schuhmann et al., âLaion-5b: An open large-scale dataset for training next generation image-text models,â NeurIPS, 2022.

[51] J. Kerr et al., âLerf: Language embedded radiance fields,â ICCV, 2023.

[50] A. Fang et al., âData filtering networks,â ICLR, 2024.

## APPENDIX

## A. Datasets

ScanNet++ contains 1752 Ã 1168 RGB-D images of real indoor scenes with ground-truth 3D meshes and instance and semantic annotations. For training, we use the top 100 semantic labels from the more than 1.6K annotated semantic classes, and evaluate on the whole set of 1.6K labels. Its training set has 230 scenes and its validation set has 50 scenes. Each scene has a training camera trajectory and an independent validation one.

ScanNetv2 also images real indoor scenes at RGB resolution of $1 2 9 6 \times 9 6 8$ and depth resolution of $6 4 0 \times 4 8 0$ . It also has ground-truth 3D meshes with ground-truth instance and semantic annotations. ScanNetv2 has two sets of annotations, the original set with 20 classes (ScanNet20), and an expanded set with 200 classes (ScanNet200) [41]. We evaluate on the 5 scenes subset used by HOV-SG [24] (HVS), and on the whole validation set of 312 scenes (FVS). Despite some overlap in physical scenes, ScanNet and ScanNet++ were captured years apart, with different trajectories and sensors, making images and reconstructions significantly different. Image blur and noisy depths make ScanNet more challenging than ScanNet++.

Replica is a synthetic dataset generated from high-fidelity real-world data. Scenes consist of ground-truth 3D meshes with semantic annotations. For all scenes, RGB-D sequences have been rendered at 1200Ã680. For Replica we use the common 8 scenes subset (office-0...4, room-0...2) with NICE-SLAM camera trajectories [43].

## B. Implementation

Our CLIP merging has a 5-layer transformer encoder with 8 heads and a 4-layer MLP. It was trained on ScanNet++ train set for 15 epochs, with batch size 512, on 4 V100 GPUs. As pre-processing, we computed segmentation masks on images, matched these with their ground-truth 2D semantic labels, and pre-computed input and target CLIP embeddings to speed up the training process.

Regarding OVO, we use the pixel size of segmented 2D masks as metric of viewpoints quality, and show results selecting the final descriptor between the 10 best keyframes of each 3D segment. Except when stated otherwise, we relied on SAM2.1-l for 2D instance segmentation, and SigLip ViT-SO400 for CLIP descriptors. We query the models with the set of classes of each dataset using the template âThis is a photo of a {class}â. For fairness in OVO evaluation, we reproduce previous approachesâ [9, 10, 23, 24] keyframes selection and querying. We select as keyframes 1 every 10 frames. The representation is queried with each datasetâs semantic classes, and each 3D segment is matched to the class with higher similarity. Following HOV-SG, the vertices of our estimated point cloud are matched to the vertices of ground-truth meshes using KD-tree search with 5 neighbors. Profiling experiments were run on Ubuntu 20, with an i7- 11700K CPU, an RTX-3090 GPU, 64 GB of RAM and 150 GB of swap.

Due to slight differences in metrics computation, we reproduced HOV-SG and Open3DIS in both Replica and ScanNetv2. For a fairer comparison with Open-3DIS we implemented it with SigLIP ViT-SO400M rather than its base CLIP ViT-L/14. We where unable to make OpenNerf converge in ScanNetv2, probably due to the impact of its noisy GT camera poses in NeRFs convergence. We report OpenNeRF official metrics on Replica. In this section we report minor ablations and experiments performed during OVOâs development using ScanNet++ training set. First we report an ablation of different foundation models for 2D instance segmentation, and language-image features extraction. Then, we ablate the algorithm to merge different CLIP descriptors and validate our proposed CLIP merging. We profit from the CLIP merging to reduce the number of CLIPs descriptors computations and evaluate the impact of the number of views on the selection of the final descriptor of 3D instances. After that, we present a mask bleeding problem that arises from depth estimation inaccuracies, and how we tackled it. Finally, we report an overall profiling of the system using different previously ablated components.

While the segmentation backbones where ablated on a single scene from ScanNet++, we used an extended set of five scenes for CLIP [8] models and similarity computation, to ablate the set of fixed weights, the evaluation of the number of viewpoints, and the mask bleeding. Then we used a different set of 10 scenes for the overall profiling to avoid overfitting on the previous set. Regarding CLIP merging training was done using the 230 scenes from ScanNet++ training set, and validation against baselines was performed on ScanNet++ 50 scenes validation set, and on ADE20K-150. We measured mean Intersection over Union (mIoU) of the 3D semantic segmentation.

As starting point, segmentation masks are computed using SAM 2 [44]; CLIP vectors are computed from masks using SigLIP-384; for each mask three vectors are computed and weighted together as introduced by HOV-SG [24]; each 3D object gets assigned the CLIP vector from the view that minimizes the L1 distance to its other views. Finally semantic classes are matched to each 3D object using the similarity approach presented by LangSplat [45].

## C. Foundation Models

a) SAM: Since its release, Segment Anything Model (SAM) [44] has been the state-of-the art for out-of-the box instance segmentation on different fields. Its segmenteverything mode extracts multiple masks from a single image, taking an input a grid of point on the image. Nevertheless, this mode has a low throughput mainly due to the postprocessing required to filter duplicated and bad segmentation masks. Although several methods claim up to Ã100 speedups with respect to SAM, these speed-ups are measured when segmenting a single object on the image, and do not measure the segment-everything mode and its post-processing.

In this ablation the evaluated models are SAM [44], SAM 2 [15], FastSAM [46], and EfficientViTSAM [47]. The evaluation in Tab. VIII shows how when segmenting everything these methods do not imply an improvement against a SAM implementation with tuned hyper-parameters.

<table><tr><td rowspan=6 colspan=1>90Â°0â 00</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=3 colspan=1>14.63</td><td rowspan=3 colspan=1>14.39</td><td rowspan=3 colspan=1>14.73</td><td rowspan=3 colspan=1>14.7</td><td rowspan=3 colspan=1>14.61</td><td rowspan=3 colspan=1>14.85</td><td rowspan=3 colspan=1>15.46</td><td rowspan=3 colspan=1>15.21</td><td rowspan=3 colspan=1>15.28</td><td rowspan=3 colspan=1>14.7</td><td rowspan=3 colspan=1>14.45</td><td rowspan=3 colspan=1>14.07</td><td rowspan=3 colspan=1>14.47</td><td rowspan=3 colspan=1>14.38</td><td></td></tr><tr><td rowspan=11 colspan=1>14.25          -15.6-15.414.45-15.2-15.014.4214.814.4          -14.6-14.414.4-14.20.6</td></tr><tr><td rowspan=1 colspan=1>14.25</td></tr><tr><td rowspan=2 colspan=1>14.83</td><td rowspan=2 colspan=1>14.63</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=1>14.4</td><td rowspan=2 colspan=1>14.66</td><td rowspan=2 colspan=1>14.42</td><td rowspan=2 colspan=1>14.45</td></tr><tr><td rowspan=1 colspan=1>14.74</td><td rowspan=1 colspan=1>14.81</td><td rowspan=1 colspan=1>14.83</td><td rowspan=1 colspan=1>15.14</td><td rowspan=1 colspan=1>15.17</td><td rowspan=1 colspan=1>15.21</td><td rowspan=1 colspan=1>15.3</td><td rowspan=1 colspan=1>14.86</td><td rowspan=1 colspan=1>14.45</td></tr><tr><td rowspan=7 colspan=1>pqoM800bLL&#x27;O</td><td rowspan=1 colspan=1>14.36</td><td rowspan=1 colspan=1>14.45</td><td rowspan=1 colspan=1>14.5</td><td rowspan=1 colspan=1>14.84</td><td rowspan=1 colspan=1>14.91</td><td rowspan=1 colspan=1>15.21</td><td rowspan=1 colspan=1>15.17</td><td rowspan=1 colspan=1>15.17</td><td rowspan=1 colspan=1>15.19</td><td rowspan=1 colspan=1>15.2</td><td rowspan=1 colspan=1>14.99</td><td rowspan=1 colspan=1>14.48</td><td rowspan=1 colspan=1>14.84</td><td rowspan=1 colspan=1>14.49</td><td rowspan=1 colspan=1>14.42</td></tr><tr><td rowspan=2 colspan=1>14.7</td><td rowspan=2 colspan=1>14.42</td><td rowspan=2 colspan=1>14.28</td><td rowspan=2 colspan=1>14.94</td><td rowspan=2 colspan=1>15.14</td><td rowspan=2 colspan=1>15.38</td><td rowspan=2 colspan=1>15.36</td><td rowspan=2 colspan=1>15.77</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=1>14.86</td><td rowspan=2 colspan=1>14.75</td><td rowspan=4 colspan=1>14.414.4</td></tr><tr><td rowspan=1 colspan=1>15.38</td><td rowspan=1 colspan=1>15.54</td><td rowspan=1 colspan=1>15.11</td><td rowspan=1 colspan=1>15.04</td></tr><tr><td rowspan=2 colspan=1>14.6</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=1>14.78</td></tr><tr><td rowspan=1 colspan=1>14.57</td><td rowspan=1 colspan=1>14.52</td><td rowspan=1 colspan=1>14.48</td><td rowspan=1 colspan=1>15.21</td><td rowspan=1 colspan=1>15.29</td><td rowspan=1 colspan=1>15.35</td><td rowspan=1 colspan=1>15.51</td><td rowspan=1 colspan=1>15.68</td><td rowspan=1 colspan=1>15.67</td><td rowspan=1 colspan=1>15.28</td><td rowspan=1 colspan=1>15.25</td><td rowspan=1 colspan=1>15.1</td></tr><tr><td rowspan=1 colspan=1>0.3</td><td rowspan=1 colspan=1>0.32</td><td rowspan=1 colspan=1>0.34</td><td rowspan=1 colspan=1>0.36</td><td rowspan=1 colspan=1>0.39</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=2>0.43  0.45</td><td rowspan=1 colspan=1>0.47</td><td rowspan=1 colspan=1>0.49</td><td rowspan=1 colspan=1>0.51</td><td rowspan=1 colspan=1>0.54</td><td rowspan=1 colspan=1>0.56</td><td rowspan=1 colspan=1>0.58</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Wmasked</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr></table>

Fig. 6. Grid search for CLIP weights merging on five scenes from ScanNet++ [38].

TABLE VIII  
SEGMENTATION BACKBONE ABLATION.
<table><tr><td rowspan="2">SAM backbone</td><td colspan="2">281bc17764</td></tr><tr><td>mIoUâ</td><td>Latency [s]â</td></tr><tr><td>FastSAM [46]</td><td>5.0</td><td>.  $0 . 4 0 \pm 0 . 2 7$ </td></tr><tr><td>EfficientViTSAM [47]</td><td>17.1</td><td> $4 . 1 9 \pm 0 . 8 5$ </td></tr><tr><td>EfficientViTSAM [47] - tuned</td><td>15.1</td><td> $0 . 6 8 \pm 0 . 0 5$ </td></tr><tr><td>SAM [44]</td><td>19.0</td><td> $5 . 4 3 \pm 1 . 8 3$ </td></tr><tr><td>SAM [44] - tuned</td><td>18.1</td><td> $0 . 8 4 \pm 0 . 1 3$ </td></tr><tr><td>SAM 2 [15] - tuned</td><td>19.1</td><td> $0 . 7 1 \pm 0 . 1 0$ </td></tr></table>

TABLE IX

CLIP ABLATION RESULTS ON 5 SCENES FROM SCANNET++.
<table><tr><td>Architecture</td><td>Resolution</td><td>mIoU [%]</td><td>Latency [s]</td></tr><tr><td rowspan="5">DFN-ViT-B-16 DFN-ViT-L/14 DFN-ViT-H/14</td><td></td><td>10.92</td><td> ${ \bf 0 . 1 0 0 \pm 0 . 0 2 2 }$ </td></tr><tr><td></td><td>11.89</td><td> $0 . 1 7 3 \pm 0 . 0 3 1$ </td></tr><tr><td> $2 2 4 \times 2 2 4$ </td><td>13.22</td><td> $0 . 2 8 6 \pm 0 . 0 5 4$ </td></tr><tr><td>12.71</td><td></td><td> $0 . 2 8 3 \pm 0 . 0 5 3$ </td></tr><tr><td>13.78</td><td> $0 . 2 2 9 \pm 0 . 0 2 6$ </td><td></td></tr><tr><td rowspan="2">SigLIP-SO400M 384 DFN-ViT-H/14-378</td><td> $3 8 4 \times 3 8 4$ </td><td>15.35</td><td> $0 . 4 4 2 \pm 0 . 0 8 0$ </td></tr><tr><td></td><td>12.96</td><td> $0 . 6 6 4 \pm 0 . 1 3 6$ </td></tr></table>

b) Visual-Language descriptors.: To compute imagelanguage features we rely on the family of CLIP and its variants. To select the CLIP architecture we evaluate the difference in performance and latency of different SOTA models to compute CLIP embeddings:

â¢ OpenCLIP [48] base ViT-H-14, trained on LAION-2B English [49] at a resolution of $2 2 4 \times 2 2 4$ , using CLIPâs cosine similarity.

â¢ DFN [50] ViT-B-16, ViT-L-14, and ViT-H-14 trained on the dataset DFN-5b [50] with input images of 224Ã224, and a ViT-H-14 finetuned at resolution 384Ã384, using CLIPâs cosine similarity.

â¢ two SigLIPâs Shape-Optimized 400M parameter ViT (ViT SO-400M), trained on WebLI English dataset at 224 Ã 224, with one fine-tuned at $3 8 4 \times 3 8 4$ , and optimized using SigLIPâs cosine similarity.

In this ablation each backbone is evaluated using the similarity with which they were trained, without ensembling, and using the template âThis is a photo of a {class}â. The results in Tab. IX show a clear trade-off between segmentation performance, and model latency. SigLIP-384 achieves the best mIoU, while SigLIP at 224 Ã 224 has the best balance between mIoU and speed. Overall, this ablation shows the importance of selecting the proper CLIP backbone, with a difference of almost 5% between the best and the worst model.

## D. CLIP descriptors merging

a) Similarity computation.: Initially, CLIP [8] presented the cosine similarity, cos $( \phi _ { \mathrm { q r y } } , \phi _ { \mathrm { i m g } } )$ to compute the distance between the text, $\phi _ { \mathrm { q r y } }$ , and image, Ïimg, embeddings. SigLIP [16] adapted it to its loss function, as Sigmoid (cos $( \phi _ { \mathrm { q r y } } , \phi _ { \mathrm { i m g } } ) \times \tau + b )$ , including a Sigmoid operation, and the learned inverse temperature, $\begin{array} { r } { t { } = { } \frac { 1 } { \tau } , } \end{array}$ and bias b parameters. To classify, both approaches assigned to an image the class of the query that generated the highest similarity. Also based on CLIPâs experiments, to compute the cosine similarity HOV-SG [24] computed query embeddings as $\begin{array} { l c l } { { \phi _ { \mathbf { q r y } } } } & { { = } } & { { \frac { \phi _ { \mathrm { c l } } ^ { - } + \phi _ { \mathrm { t e m p } } } { 2 } } } \end{array}$ , where $\phi _ { c l }$ is the text embedding computed from the class name, and $\phi _ { \mathrm { t e m p } }$ is the text embedding computed from the phrase resulting of inserting the class into the template âThere is {class} in the sceneâ. In contrast, LERF [51] proposed to compute the cosine similarity between the image and text embeddings as

$$
\operatorname* { m i n } _ { i } \frac { \exp \big ( \cos ( \phi _ { \mathrm { q r y } } , \phi _ { \mathrm { i m g } } ) \big ) } { \exp \big ( \cos ( \phi _ { \mathrm { q r y } } , \phi _ { \mathrm { i m g } } ) \big ) + \exp \big ( \cos ( \phi _ { \mathrm { c a n } } ^ { i } , \phi _ { \mathrm { i m g } } ) \big ) } ,\tag{2}
$$

where $\phi _ { \mathrm { c a n } } ^ { i }$ is the text embedding of one of the predefined canonical queries object, things, stuff, texture.

Using the SigLIP ViT-SO400M model to compute CLIP vectors, we compare between:

<!-- image-->  
Fig. 7. Evaluation using only top views to compute CLIP on 5 scenes from ScanNet++ [38]. While using more than one view has substantial impact on the runtime, it also improves segmentation accuracy. However, too many views also degrade the segmentation accuracy.

TABLE X  
SIMILARITY COMPUTATION ABLATION ON 5 SCENES FROM SCANNET++ MEASURING SEMANTIC 3D MIOU.
<table><tr><td></td><td>Cosine similarity</td><td>LERF&#x27;s similarity</td></tr><tr><td>w ensemble</td><td>14.75%</td><td>14.75%</td></tr><tr><td>w\o ensemble</td><td>15.35%</td><td>14.98%</td></tr></table>

â¢ computing query embeddings, $\phi _ { \mathrm { q r y } } ,$ only with the template âThis is a photo of a {class}â or as an ensemble averaging the template embedding with the class embedding;

â¢ and computing SigLIPâs cosine similarity or LERFâs cosine similarity.

Results in Tab. X show how the basic configuration of using SigLIP similarity without ensemble achieves the best performance. From here on, all experiments will proceed using basic cosine similarity without ensemble. To focus CLIP descriptors to elements in an image, we follow HOV-SGâs [24] approach. For each mask segmented by SAM, HOV-SG proposed to compute CLIP embeddings combining the information of the complete image, the masked image without background, and a bounding box of the mask including background. For each segmentation mask i, its corresponding CLIP vector $F _ { i }$ is computed as

$$
F _ { i } = F _ { \mathrm { g l o b a l } } \times w _ { \mathrm { g l o b a l } } + F _ { \mathrm { l o c a l } _ { i } } \times ( 1 - w _ { \mathrm { g l o b a l } } ) ,\tag{3}
$$

with

$$
F _ { \mathrm { l o c a l } _ { i } } = F _ { \mathrm { m a s k e d } _ { i } } \times w _ { \mathrm { m a s k e d } } + F _ { \mathrm { b b o x } _ { i } } \times ( 1 - w _ { \mathrm { m a s k e d } } ) ,\tag{4}
$$

combinig the CLIP vector of the whole image, $F _ { \mathrm { g l o b a l } }$ the CLIP vector of only the segmentation mask without background, $F _ { \mathrm { m a s k e d } _ { i } }$ , and the one of the bounding box of the segmentation mask including background, $F _ { \mathrm { b b o x } _ { i } }$

HOV-SG [24] used

$$
w _ { \mathrm { g l o b a l } } = \operatorname { S o f t m a x } ( \cos ( F _ { \mathrm { g l o b a l } } , F _ { i } ) ) ,\tag{5}
$$

and $w _ { \mathrm { m a s k e d } } = 0 . 4 4 1 8$ . Nevertheless, the use of the Softmax introduced a dependency between the different embeddings extracted on the same frame. To avoid computing all CLIP embeddings on every frame, we remove the Softmax and perform a grid search of $w _ { \mathrm { m a s k e d } }$ and $w _ { \mathrm { g l o b a l } }$ . The best performance is achieved for $w _ { \mathrm { g l o b a l } } = 0 . 4 5$ and $w _ { \mathrm { m a s k e d } } = 0 . 0 9 7 5$ as shown in Fig. 6.

b) CLIP merging: Rather than relying on 3 fixedweights that ideally should be tunned for each scene, we developed the CLIP merging to estimate the corresponding weight for each image. After training on ScanNet++ train set with the top 100 semantic labels, we evaluate its performance on the ScanNet++ validation set using the total set of 1.6k queries, both including (w.top 100) and excluding (w/o. top 100) classes seen during training. For a stronger distribution switch, we also evaluate on ADE20k-150.

Comparing its performance against HOV-SGâs approach and our variation of HOV-SGâs using three fixed weights, the CLIP merging outperforms the baselines using all the labels, Tab. XI. Excluding from the metrics the 100 labels seen during training, we can observe how the CLIP merging performance drops with respect to the baselines. Despite the slight bias toward classes at training, it still outperform on freq. weighted metrics of classes that werenât seen during training, and on novel data on the ADE20k-150 dataset.

Although, OVO-mapping evaluation in Replica and Scan-Netv2, additional segmentation metrics on classes outside the training set (Tab. XIII showcase how the bias does not have an impact on our CLIP mergingâs generalization. Our method accurately detects in 3D several unseen classes across Replica and ScanNetv2, including guitar, coffee maker, blackboard, and scale. The mIoU for these examples exceeds 60%. From here on, all experiments will proceed using the CLIP merging.

TABLE XI  
OUR CLIP MERGING VS. BASELINES ON: SCANNET++ (S++) USING 1.6K QUERIES (METRICS ON OBSERVED 495 LABELS, W. AND W/O. THE TOP 100 USED AT TRAINING), AND ADE20K WITH 150 LABELS. COLOR INDICATES FIRST , SECOND , AND THIRD BEST.
<table><tr><td></td><td></td><td>S++ w. top 100</td><td></td><td></td><td></td><td>S++ w/o. top100</td><td></td><td></td><td></td><td>ADE20k-150</td><td></td><td></td></tr><tr><td>Method</td><td>mIoU</td><td>mAcc</td><td>f-mIoU</td><td>f-mAcc</td><td>mIoU</td><td>mAcc</td><td>f-mIoU</td><td>f-mAcc</td><td>mIoU</td><td>mAcc</td><td>f-mIoU</td><td>f-mAcc</td></tr><tr><td>HOV-SG</td><td>9.4</td><td>15.9</td><td>12.8</td><td>15.9</td><td>8.3</td><td>15.1</td><td>8.4</td><td>13.6</td><td>21.9</td><td>53.7</td><td>22.3</td><td>34.9</td></tr><tr><td>Fixed-weights</td><td>9.4</td><td>15.9</td><td>13.1</td><td>16.3</td><td>8.3</td><td>15.1</td><td>8.4</td><td>13.8</td><td>22.4</td><td>53.9</td><td>23.1</td><td>35.5</td></tr><tr><td>CLIP-merger</td><td>10.7</td><td>16.9</td><td>36.1</td><td>45.3</td><td>7.3</td><td>12.8</td><td>9.9</td><td>15.0</td><td>23.4</td><td>49.3</td><td>28.7</td><td>41.2</td></tr></table>

TABLE XII

AVERAGE RUNTIMES AND 3D SEMANTIC PERFORMANCE ON SCANNET++. WE MEASURE THE SEGMENTATION (SEG); SEGMENTS MATCHING AND TRACKING (M&T); SEGMENTS PRE PROCESSING (PP); CLIPS COMPUTATION (CLIP); AND TOTAL SECONDS PER KEY FRAME (s/KF ).
<table><tr><td>CLIP</td><td>SAM</td><td># best views</td><td>Seg. [s]</td><td>M&amp;T [s]</td><td>PP []</td><td>CLIP [s]</td><td>s/KF</td><td>mIoU</td><td>mAcc</td><td>f-mIoU</td><td>f-mAcc</td></tr><tr><td rowspan="2">ViT-H/14</td><td>1-H 2.1-L</td><td rowspan="2">10</td><td>1.516</td><td>0.269</td><td>0.085</td><td>0.175</td><td>2.112</td><td>13.3</td><td>22.4</td><td>20.2</td><td>31.7</td></tr><tr><td></td><td>0.338</td><td>0.252</td><td>0.066</td><td>0.135</td><td>0.865</td><td>14.1</td><td>24.9</td><td>27.3</td><td>37.7</td></tr><tr><td rowspan="3">SigLIP</td><td>2.1-t</td><td>10</td><td>0.245</td><td>0.247</td><td>0.057</td><td>0.204</td><td>0.820</td><td>11.8</td><td>25.7</td><td>34.2</td><td>46.6</td></tr><tr><td>2.1-L</td><td></td><td>0.339</td><td>0.253</td><td>0.05</td><td>0.233</td><td>00.957</td><td>14.2</td><td>27.0</td><td>34.3</td><td>45.6</td></tr><tr><td></td><td>all</td><td>0.337</td><td>0.261</td><td>0.110</td><td>0.367</td><td>1.167</td><td>15.8</td><td>29.6</td><td>36.3</td><td>48.6</td></tr></table>

TABLE XIII

CLIP MERGING GENERALIZATION. 3D METRICS ON SCANNETV2 OF SOME CLASSES NOT SEEN DURING TRAINING.
<table><tr><td></td><td>scale</td><td>toaster oven</td><td>blackboard</td><td>coffee maker</td><td>guitar</td><td>projector screen</td></tr><tr><td>mIoU%</td><td>75.1</td><td>78.53</td><td>61.4</td><td>67.0</td><td>62.68</td><td>64.1</td></tr><tr><td>mAcc%</td><td>81.2</td><td>94.07</td><td>76.1</td><td>86.7</td><td>86.79</td><td>86.8</td></tr></table>

## E. Additional heuristics

a) NÂº of best views.: To reduce the expensive CLIP computation for each frame, we evaluate the impact of using only the best views where each 3D segment has been seen to compute its CLIP descriptor. We evaluate from using only the best image to using all the images where the object has been seen. The quality of an image is based on the area of the objectâs 2D segmentation in it.

For a sequence of 51 keyframes, we evaluate for k â {1, . . . , 51}, being all using all the views to compute objects 3D vectors. The results show, see Fig. 7, that neither using only the best nor using all the views are robust enough to noise. For the set of 5 scenes on this experiment, the best values of k are between 2 and 7, achieving an mIoU around 18%, almost 3 points better than using all observations, although, the perfect value of will probably be scene and object dependent. We decide to set use 10 views as a balance to avoid useless computation of CLIP vectors and being resistant to noisy images.

b) Masks bleeding.: Observing OVO-SLAM matching results, we noticed some problems related with SAMâs masks. When some 3D points are projected on the edges of a 2D mask to which they do not belong, they are wrongly clustered into it and matched to a 3D instance. Then, when these are seen again they will propagate the wrongly assigned ID. This phenomenon can be observed in particular on the edges of objects, where the depth and masks are less accurate, and masks propagate the ID of the object to the background, as seen in Fig. 8. To compensate it we developed two approaches:

â¢ First, we add a filter to only keep matches of 3D points that are assigned to the same object in two consecutive frames;

â¢ Second, we apply a low-pass filter to the depth map to mask the edges of the objects and avoid matching points around them.

Results on Tab. XIV show how while using the depth filter does improve the average mIoU, the limitation to match in consecutive frames does not. As a consequence we keep only the depth filter although it does not completely solve the problem.

TABLE XIV  
MASK BLEEDING SOLUTIONSâ ABLATION ON 5 SCENES FROM SCANNET++ [38].
<table><tr><td>Config</td><td>mIoUâ</td></tr><tr><td>Base</td><td>15.80%</td></tr><tr><td>w depth filter</td><td>16.16%</td></tr><tr><td>w consecutive KF filter</td><td>15.07%</td></tr><tr><td>w both</td><td>15.82%</td></tr></table>

c) Overall profiling.: Finally, we quantify the latencyquality trade-off in our architecture evaluating selected foundation models and number of views against less powerful alternatives. This evaluation is performed on a different set of 10 scenes from ScanNet++ to avoid over-fitting to the previous 5 scenes. For 2D segmentation we evaluate SAM [44] with ViT-H/14 encoder (1-H), and SAM 2.1 [15] with Hiera large (2.1-L) and Hiera tiny (2.1-t) image encoders. For CLIP extraction, we evaluate DFN ViT-H/14-378 [50] and SigLIP-SO400 [16] both with input images of 384 pixels. The results in Tab. XII show that in this set of scenes the best 3D segmentation is achieved with the largest models using all points of view. Nevertheless, the best trade-off can be achieved reducing the number of views and the CLIP model.

<!-- image-->  
Fig. 8. Mask bleeding and propagation produced by masks inaccuracy. The edges of the chair (pink) bleed to the background at $k _ { n } ,$ and therefore the segment label is wrongly propagated to the it in the following keyframes.