# RadarGaussianDet3D: Gaussian Representation-based Real-time 3D Object Detection with 4D Automotive Radars

Weiyi Xiong, Bing Zhu, Senior Member, IEEE, and Zewei Zheng, Member, IEEE

Abstractâ4D automotive radars have gained increasing attention for autonomous driving due to their low cost, robustness, and inherent velocity measurement capability. However, existing 4D radar-based 3D detectors rely heavily on pillar encoders for BEV feature extraction, where each point contributes to only a single BEV grid, resulting in sparse feature maps and degraded representation quality. In addition, they also optimize bounding box attributes independently, leading to suboptimal detection accuracy. Moreover, their inference speed, while sufficient for high-end GPUs, may fail to meet the realtime requirement on vehicle-mounted embedded devices. To overcome these limitations, an efficient and effective Gaussianbased 3D detector, namely RadarGaussianDet3D is introduced, leveraging Gaussian primitives and distributions as intermediate representations for radar points and bounding boxes. In RadarGaussianDet3D, a novel Point Gaussian Encoder (PGE) is designed to transform each point into a Gaussian primitive after feature aggregation and employs the 3D Gaussian Splatting (3DGS) technique for BEV rasterization, yielding denser feature maps. PGE exhibits exceptionally low latency, owing to the optimized algorithm for point feature aggregation and fast rendering of 3DGS. In addition, a new Box Gaussian Loss (BGL) is proposed, which converts bounding boxes into 3D Gaussian distributions and measures their distance to enable more comprehensive and consistent optimization. Extensive experiments on TJ4DRadSet and View-of-Delft demonstrate that RadarGaussianDet3D achieves high detection accuracy while delivering substantially faster inference, highlighting its potential for realtime deployment in autonomous driving. Source code is available at https://github.com/XiongWeiyi/RadarGaussianDet3D.

Index Termsâ4D imaging radar, 3D Gaussian splatting, 3D object detection, deep learning, autonomous driving.

## I. INTRODUCTION

A CCURATE and fast perception is essential for safety-critical autonomous driving. Since autonomous vehicles may operate under diverse conditions such as heavy rain or darkness, sensors capable of functioning reliably in all environments are necessary. Unlike cameras and LiDARs, which rely on visible or invisible light, automotive radars employ millimeter waves that are robust against poor lighting and extreme weather, making them indispensable. Furthermore, radars capture velocity information, which is crucial for understanding dynamic scenes. Although conventional radars lack elevation measurement, preventing 3D environmental perception, this limitation has been overcome with the development of 4D radars. In addition to enabling height sensing, 4D radars provide higher-resolution data, which enhances object identification and localization [1].

<!-- image-->  
Fig. 1. Illustration of different point scattering methods. Purple arrows from a point indicate the BEV grid cells influenced by that point. (a) Pillar scatter in PointPillars [3] maps each point to a single grid cell based on coordinates. (b) Enhanced scatter methods, such as the RCS-aware scatter in RCBEVDet [4], define a neighborhood for each point and assign it to all grid cells whose centers lie within the neighborhood. (c) Gaussian splatting in the proposed RadarGaussianDet3D rasterizes point-converted Gaussian primitives onto the BEV plane, allowing each point to contribute to all overlapping grid cells.

Despite these advantages, 4D radars also suffer from clear drawbacks, namely data sparsity and noise. Compared with LiDARs, the resolution of 4D radars remains significantly lower [1], resulting in sparse radar point clouds. Noise is also easily introduced due to factors such as the multi-path effect and receiver saturation [2], making precise perception difficult.

Within the perception framework, accurate and robust 3D object detection serves as a cornerstone for the subsequent trajectory prediction and motion planning in autonomous driving. This paper addresses three major challenges encountered by existing 4D radar-based 3D detectors. First, most detectors adopt the pillar encoder in PointPillars [3] to construct radar birdâs-eye-view (BEV) feature maps. As noted by Lin et al. [4] and shown in Fig. 1(a), the scatter operation in the pillar encoder maps sparse radar points to individual BEV grids based on their coordinates, producing sparse feature maps. Although stacking additional BEV encoder layers can mitigate this problem, small object features may blend into background regions. RCBEVDet [4] proposes an enhanced scatter operation that assigns each point feature to all neighboring grids (Fig. 1(b)), with neighborhood size determined by the radar cross section (RCS), which roughly reflects object size. However, this hand-crafted design is sub-optimal, since RCS is influenced not only by object size but also by factors such as material properties. In addition, if a grid lies within the neighborhoods of multiple points, only the maximum feature value is retained, leading to information loss. Therefore, a more suitable feature scattering strategy must be explored.

Second, existing bounding box regression losses (e.g., L1 loss) typically measure the difference of each box attribute separately, neglecting correlations among them. For instance, box localization error is usually computed without considering size or orientation, despite the fact that the same position error may have different effects depending on these factors. Thus, a new loss function that jointly optimizes all bounding box attributes is required.

Third, autonomous vehicles must perceive and interpret their surroundings as quickly as possible to respond to potential hazards. Although most 4D radar-based detectors achieve realtime performance on high-end GPUs, they may not meet realtime requirements on vehicle-mounted embedded devices with limited computational resources [5], hindering deployment. Hence, developing effective detectors with improved runtime performance remains imperative.

To address these issues, RadarGaussianDet3D is proposed as an efficient and effective 4D radar-based 3D object detector that employs Gaussian primitives and distributions as intermediate representations of radar points and bounding boxes. For the first problem, the Point Gaussian Encoder (PGE) is designed to generate BEV feature maps using the 3D Gaussian Splatting (3DGS) technique [6]. 3DGS represents a scene with 3D Gaussian primitives, where a view is synthesized by projecting them onto a plane followed by differentiable rasterization. Since a 3D Gaussian primitive can be regarded as a point with additional attributes such as size and rotation, it is natural to predict these attributes from radar points. Note that, unlike the original 3DGS framework that requires per-scene optimization, this process is feed-forward, single-shot, and generalizable. The BEV feature map is then obtained through Gaussian splatting, which acts as a feature scattering strategy (Fig. 1(c)). In this process, each point contributes to all grids overlapped by the projected 2D Gaussian primitive, and alphablending is used to integrate features from multiple points. For bounding box regression, a Box Gaussian Loss (BGL) is proposed, which transforms predicted and ground-truth boxes into 3D Gaussian distributions and computes their distance, thereby comprehensively optimizing all box attributes. Finally, since 3DGS [6] provides ultra-fast rendering and BGL is used only during training, the proposed model achieves high inference efficiency.

The contributions of this work are summarized as follows:

â¢ Radar points are modeled as 3D Gaussian primitives, and a Point Gaussian Encoder (PGE) is designed to efficiently splat these primitives into BEV feature maps, effectively alleviating the inherent sparsity of radar point clouds.

â¢ 3D bounding boxes are formulated as Gaussian distributions, and a Box Gaussian Loss (BGL) is introduced to measure box discrepancies via distributional distance, enabling joint optimization of all bounding box attributes.

â¢ By integrating PGE and BGL, a 4D radar-based 3D object detection framework, termed RadarGaussianDet3D, is developed. Experiments on the TJ4DRadSet [7] and View-of-Delft [8] datasets demonstrate a favorable accuracyâefficiency trade-off, achieving substantially lower latency than state-of-the-art methods while maintaining high detection accuracy.

This paper is organized as follows. Section II reviews related work, including 4D radar-based 3D object detection methods and applications of 3D Gaussian splatting in autonomous driving. Section III presents the proposed RadarGaussianDet3D model in detail, and Section IV reports and analyzes the experimental results. Finally, Section V concludes the paper.

## II. RELATED WORK

## A. 3D Object Detection with 4D Radar Point Clouds

Represented as 3D point clouds, 4D radar data share certain characteristics with LiDAR data, and thus LiDAR-based detectors such as PointPillars [3] can be directly applied [8]. However, this direct adaptation is sub-optimal because of the modality differences, requiring tailored modifications.

Several approaches focus on enriching radar features [5], [9]â[11]. For instance, RCFusion [10] processes spatial, velocity, and intensity values separately to avoid feature confusion. RadarPillars [5] introduces self-attention among non-empty pillars to aggregate features belonging to the same object. SMURF [11] predicts point density distributions using kernel density estimation as an additional feature, improving sparsityawareness in detection.

Another line of work addresses the inherent deficiencies of 4D radar. RCBEVDet [4] defines a neighborhood for each point and scatters features across all BEV grids in the area, thereby densifying BEV feature maps and mitigating point cloud sparsity. MAFF-Net [12] proposes a cylindrical denoising assist module to identify keypoints around objects, reducing the effect of noise.

As certain limitations of 4D radar cannot be fully resolved algorithmically, some methods use auxiliary modalities during training. E.g., SCKD [13] employs cross-modality distillation to transfer knowledge from LiDAR to 4D radar, enhancing performance without compromising efficiency. Other approaches explore multi-modal fusion, integrating geometric information from LiDAR point clouds [14]â[16] or semantic cues from RGB images [17]â[20] to complement 4D radar.

In contrast to the aforementioned methods, this work addresses the sparsity of 4D radar point clouds at the representation level. Instead of relying on the commonly used pillarbased representation, radar points are modeled as 3D Gaussian primitives, and these primitives are rasterized using 3DGS to generate denser BEV feature maps.

## B. 3D Gaussian Splatting in Autonomous Driving

3D Gaussian Splatting (3DGS) [6] introduced a new paradigm in 3D reconstruction. Although initially proposed for camera-based novel view synthesis (NVS) in static scenes, researchers extended it to other domains, including autonomous driving. For instance, SplatAD [21] and RadarSplat [2] extend 3DGS to NVS with LiDAR and radar sensors, respectively, while DrivingGaussian [22] adapts it to dynamic scenes.

3DGS has also been applied to various perception tasks, including BEV segmentation [23], [24], 3D object detection [25], and occupancy prediction [26]â[28]. However, perception tasks typically require strong generalization capability and real-time inference, making per-scene optimization of Gaussian attributes in the original 3DGS formulation impractical.

<!-- image-->  
Fig. 2. Overview of RadarGaussianDet3D.

As a result, existing perception-oriented methods predict Gaussian attributes in a feed-forward manner, either directly from feature maps [23], [24], [29] or via learnable queries [26], [27]. In the feature-based paradigm, each Gaussian primitive is predicted directly from a feature map element (e.g., an image pixel), resulting in a simple and efficient design that is particularly suitable for single-modality inputs. Moreover, this one-shot prediction incurs minimal computational overhead, making it highly favorable for real-time systems. In contrast, query-based methods rely on iterative refinement of Gaussian primitives through repeated query-data interactions. While this design allows a variable number of Gaussian primitives and naturally supports multi-modal inputs, it introduces additional computational cost. Beyond perception subtasks, query-based Gaussian representations have also been extended to end-toend autonomous driving [30], [31] and world models [28].

Once Gaussian primitives are predicted, they are splatted onto the BEV plane or rasterized in the 3D space, producing BEV pseudo-images or voxel grids that are subsequently processed by task-specific heads.

While existing approaches have achieved promising results, most focus on image-view transformation or multi-modal feature fusion. In contrast, this work pioneers the application of 3DGS to 4D radar point clouds for BEV feature extraction, where Gaussian attributes are directly predicted from sparse radar measurements. This design establishes a new paradigm for leveraging 3DGS in radar-based perception tasks.

## III. PROPOSED METHOD

## A. Overview

Fig. 2 presents the overall framework of the proposed RadarGaussianDet3D model. Unlike most existing methods that rely on pillarization, RadarGaussianDet3D employs the concept of 3DGS to construct radar BEV feature maps through a dedicated Point Gaussian Encoder (PGE). After aggregating both local and global features, the PGE predicts Gaussian attributes for each radar point, which are subsequently splatted onto the BEV plane. The resulting feature map is then processed by the backbone, neck, and detection head to produce 3D bounding box predictions.

In addition, an auxiliary loss termed Box Gaussian Loss (BGL) is introduced during training. Ground-truth and predicted bounding boxes are transformed into 3D Gaussian distributions, and their distributional distance is computed. This formulation guides the network to comprehensively optimize all bounding box attributes in a unified manner.

## B. Point Gaussian Encoder (PGE)

The PGE consists of four main steps: Local Feature Aggregation (LFA), Global Feature Aggregation (GFA), Gaussian attribute prediction, and BEV Gaussian splatting.

Local Feature Aggregation (LFA). The most straightforward approach to transform a point into a Gaussian primitive is to directly predict Gaussian attributes. However, raw radar points contain limited information and are insufficient to support reliable prediction. Therefore, local features are aggregated to enrich the representation of each point.

For the i-th point, a lightweight PointNet [32] is applied to all points in its spherical neighborhood ${ \mathcal { N } } _ { i } \colon$

$$
f _ { \mathrm { L F A } } ^ { i } = \frac { 1 } { \lvert \mathcal { N } _ { i } \rvert } \sum _ { j \in \mathcal { N } _ { i } } \mathrm { L i n e a r } \big ( \mathrm { C o n c a t } ( [ f _ { j } , p _ { j } - p _ { i } ] ) \big ) ,\tag{1}
$$

where $p _ { i } \in \mathbb { R } ^ { 3 }$ and $f _ { i } \in \mathbb { R } ^ { C _ { \mathrm { r a w } } }$ denote the 3D coordinates and raw features of the i-th point $( i = 1 , \cdots , N )$ , respectively. Here, $f _ { \mathrm { L F A } } ^ { i } \in \mathbb { R } ^ { C }$ represents the updated features after LFA. N denotes the total number of points, and C refers to the channel dimension. The i-th point is referred as the center point, while points in ${ \mathcal { N } } _ { i }$ are its neighbors. In addition, the position offset from the center point to each neighbor is appended to features, following the pillar feature extraction in PointPillars [3].

To avoid traversing all points for averaging neighbor features and to reduce memory usage, an optimized algorithm is devised. It relies on the PyTorch-supported operation scatter reduce, as visualized in Fig. 3(c). Experiments in Section IV-C confirm that the proposed Indexing & Scattering algorithm achieves higher speed than the Traversal method in Fig. 3(a), while requiring considerably less memory compared with the Broadcasting & Masking strategy in Fig. 3(b).

Global Feature Aggregation (GFA). Previous studies [5], [12] have shown that self-attention among non-empty pillars improves detection performance. Inspired by this, selfattention is applied directly among points to capture global interactions in parallel with LFA. Since radar points are sparse and most non-empty pillars contain only a single point, the computational complexity of point-based attention is comparable to that of pillar-based attention.

<!-- image-->  
Fig. 3. Visualization of different LFA implementations. For simplicity, concatenation with position offsets is omitted.

The GFA process is defined as Eq. (2):

$$
\begin{array} { r l } & { f _ { 1 } = \mathtt { L i n e a r } ( f ) , } \\ & { Q , K , V = \mathtt { M L P } \big ( \mathtt { L a y e r N o r m } ( f _ { 1 } ) \big ) , } \\ & { \quad \quad \quad f _ { 2 } = \mathtt { S e l f A t t n } ( Q , K , V ) + f _ { 1 } , } \\ & { \quad \quad f _ { \mathtt { G F A } } = \mathtt { F F N } \big ( \mathtt { L a y e r N o r m } ( f _ { 2 } ) \big ) + f _ { 2 } , } \end{array}\tag{2}
$$

where $\begin{array} { r } { f ~ \in ~ \mathbb { R } ^ { N \times C _ { \mathrm { r a w } } } } \end{array}$ denotes the raw point features, and $f _ { \mathrm { G F A } } ~ \in ~ \mathbb { R } ^ { N \times C }$ represents the features after GFA. Here, SelfAttn $( Q , K , V )$ is the self-attention operation [33] with queries Q, keys K, and values V , while FFN(Â·) denotes the feed-forward network.

Gaussian Attribute Prediction. In the original 3DGS [6], each Gaussian primitive is represented as $g = ( \mu , s , q , o , s h )$ where $\mu \in \mathbb { R } ^ { 3 } , \ s \in \mathbb { R } ^ { 3 } , \ q \in \mathbb { R } ^ { 4 } , \ o \in \mathbb { R } ,$ and $s h \in \mathbb { R } ^ { M }$ denote the 3D position (mean), 3D scales, rotation quaternion, opacity, and spherical harmonic (SH) coefficients, respectively. The SH coefficients are used to compute view-dependent colors c. Since the objective in PGE is to obtain feature maps rather than images, the SH coefficients sh are replaced with view-independent features $f ^ { g } \in \mathbb { R } ^ { C }$ . Moreover, the opacities and positions are not predicted. Specifically, opacity o is set to 1, and the position offset is fixed to 0 (i.e., $\mu ~ = ~ p )$ as introducing excessive degrees of freedom for Gaussian primitives at an early stage may hinder learning.

Gaussian attributes for each radar point are obtained by concatenating the raw features with the enriched features produced by LFA and GFA, followed by a linear projection:

$$
\begin{array} { c } { { s ^ { \prime } , q , f ^ { g } = \tt L i n e a r ( C o n c a t ( \tt [ } f , f _ { \tt L F A } , f _ { \tt G F A } ] ) ) , }  \\ { { s = \tt S i g m o i d ( } s ^ { \prime } ) \cdot s _ { \mathrm { m a x } } , \qquad } \end{array}\tag{3}
$$

where Sigmoid ensures positive scale values while preventing excessive magnitude following [34], and $s _ { \operatorname* { m a x } } ~ = ~ 1$ m is a predefined maximum scale. q is normalized to satisfy the unitnorm constraint of quaternions.

Compared with original 3DGS paradigm, where Gaussian attributes are treated as learnable parameters optimized per scene, the proposed feed-forward prediction strategy ensures model generalizability across scenes. Moreover, the single-shot prediction is computationally more efficient than the iterative refinement in query-based Gaussian prediction methods.

BEV Gaussian Splatting. In this step, the set of predicted Gaussian primitives $\mathcal { G } ~ = ~ \{ g _ { i } ~ = ~ ( \mu _ { i } , s _ { i } , q _ { i } , o _ { i } , f _ { i } ^ { g } ) \} _ { i = 1 } ^ { N }$ is splatted onto the BEV plane, with the implementation adapted from the CUDA code of 3DGS [6].

Two main modifications are introduced in the 3D-to-2D projection and rendering. First, unlike perspective projection onto the image plane, BEV projection is a parallel projection with scaling. Given the target BEV resolution (H, W ) and the BEV range $( x , y ) \in [ x _ { \operatorname* { m i n } } , x _ { \operatorname* { m a x } } ] \times [ y _ { \operatorname* { m i n } } , y _ { \operatorname* { m a x } } ]$ , the 2D mean $\mu _ { \mathrm { 2 D } }$ and 2D covariance matrix $\Sigma _ { \mathrm { 2 D } }$ are computed as

$$
\begin{array} { r } { \mu _ { \mathrm { 2 D } } = \mathbf { M } \mu , \quad \pmb { \Sigma } _ { \mathrm { 2 D } } = \mathbf { M } \pmb { \Sigma } \mathbf { M } ^ { \top } , } \\ { \mathbf { M } = [ \frac { W } { x _ { \mathrm { m a x } } - x _ { \mathrm { m i n } } } \qquad 0 \qquad 0 ] , } \\ { 0 \qquad \frac { H } { y _ { \mathrm { m a x } } - y _ { \mathrm { m i n } } } \quad 0 ] , } \end{array}\tag{4}
$$

where $\mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } = \mathbf { R S S } ^ { \top } \mathbf { R } ^ { \top } , \mathbf { \ S } = \mathbf { d i a g } ( s )$ , and R is the rotation matrix derived from quaternion q.

Second, the rendering equations of 3DGS [6] are modified by replacing the color c with Gaussian features $f ^ { g }$ , yielding the BEV feature map $F \in \mathbb { R } ^ { C \times H \times W }$

$$
\begin{array} { l } { { \displaystyle { \cal F } [ { \bf p } ] = \sum _ { i = 1 } ^ { N _ { \mathbf { p } } } f _ { i } ^ { g } \alpha _ { i } T _ { i } } , ~ T _ { i } = \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) , }  \\ { { \displaystyle \alpha _ { i } = o _ { i } \exp \left( - \frac { 1 } { 2 } ( { \bf p } - \mu _ { \mathrm { 2 D } } ) ^ { \top } \pmb { \Sigma } _ { \mathrm { 2 D } } ^ { - 1 } ( { \bf p } - \mu _ { \mathrm { 2 D } } ) \right) , } } \end{array}\tag{5}
$$

where $F [ \mathbf { p } ] \in \mathbb { R } ^ { C }$ denotes the feature vector at pixel p, and $N _ { \mathbf { p } }$ is the number of Gaussian primitives overlapping with the tile containing p.

Compared with the conventional pillar encoder that assigns each point to a single BEV grid, the proposed PGE allows each point to influence multiple neighboring grids adaptively, thereby generating denser BEV feature maps.

## C. Box Gaussian Loss (BGL)

To measure differences between bounding boxes more comprehensively, we propose BGL, a loss function that represents 3D bounding boxes as 3D Gaussian distributions and computes the distance between them.

Box-to-Gaussian Conversion. Since 3D bounding boxes and 3D Gaussian distributions share attributes ( e.g., position, scale, and orientation), they can be mutually converted. Each ground-truth bounding box $\boldsymbol { b } ~ = ~ \left[ x , y , z , l , w , h , \theta \right]$ is transformed into 3D Gaussian distribution $g = \mathcal { N } ( \mu ; \Sigma )$ as follows:

TABLE I  
RESULTS ON TJ4DRADSET T E S T SET
<table><tr><td rowspan="2">Models</td><td colspan="5">3D AP (%)</td><td colspan="5">BEV AP (%)</td><td rowspan="2">FPS (Hz)</td></tr><tr><td>Car</td><td>Ped.</td><td> ${ \mathrm { C y c . } }$ </td><td>Tru.</td><td>mAP</td><td>Car</td><td>Ped.</td><td> ${ \mathrm { C y c . } }$ </td><td>Tru.</td><td>mAP</td></tr><tr><td>PointPillars* (CVPR&#x27;19) [3]</td><td>21.26</td><td>28.33</td><td>52.47</td><td>11.18</td><td>28.31</td><td>38.34</td><td>32.26</td><td>56.11</td><td>18.19</td><td>36.23</td><td>42.9</td></tr><tr><td>CenterPoint* (CVPR&#x27;21) [35]</td><td>22.03</td><td>25.02</td><td>53.32</td><td>15.92</td><td>29.07</td><td>33.03</td><td>27.87</td><td>58.74</td><td>26.09</td><td>36.18</td><td>34.5</td></tr><tr><td>RPFA-Net* (ITSC&#x27;21) [9]</td><td>26.89</td><td>27.36</td><td>50.95</td><td>14.46</td><td>29.91</td><td>42.89</td><td>29.81</td><td>57.09</td><td>25.98</td><td>38.94</td><td>-</td></tr><tr><td>PillarNeXt* (CVPR&#x27;23) [36]</td><td>22.33</td><td>23.48</td><td>53.01</td><td>17.99</td><td>29.20</td><td>36.84</td><td>25.17</td><td>57.07</td><td>23.76</td><td>35.71</td><td>28.0</td></tr><tr><td>SMURF* (T-IV&#x27;23) [11]</td><td>28.47</td><td>26.22</td><td>54.61</td><td>22.64</td><td>32.99</td><td>43.13</td><td>29.19</td><td>58.81</td><td>32.80</td><td>40.98</td><td>23.1</td></tr><tr><td>MUFASA (ICANN&#x27;24) [37]</td><td>23.56</td><td>23.70</td><td>48.39</td><td>25.25</td><td>30.23</td><td>41.25</td><td>24.54</td><td>53.64</td><td>36.97</td><td>39.10</td><td>-</td></tr><tr><td>MAFF-Net (RAL&#x27;25) [12]</td><td>27.31</td><td>33.13</td><td>54.35</td><td>26.71</td><td>35.38</td><td>39.05</td><td>35.25</td><td>56.35</td><td>35.73</td><td>41.59</td><td> $1 7 . 9 ^ { \dagger }$ </td></tr><tr><td>RadarGaussianDet3D (Ours)</td><td>26.69</td><td>28.18</td><td>65.84</td><td>19.63</td><td>35.08</td><td>41.66</td><td>30.28</td><td>69.62</td><td>26.35</td><td>41.98</td><td>43.5</td></tr></table>

1 In each column, the highest value is in bold, the 2nd highest underlined, and the 3rd highest in italic.  
2 Inference speeds marked with â  are measured on RTX 4090, while others are on Tesla V100.  
3 The results of models marked with â are from [11], while others from corresponding citations.

$$
\mu = [ x , y , z ] , ~ \Sigma = { \bf R S S } ^ { \top } { \bf R } ^ { \top } ,\tag{6}
$$

where

$$
\begin{array} { l } { { \displaystyle { \bf S } = \mathrm { d i a g } ( [ \frac { l } { 2 a } , \frac { w } { 2 a } , \frac { h } { 2 a } ] ) = \frac { 1 } { a } \mathrm { d i a g } ( [ \frac { l } { 2 } , \frac { w } { 2 } , \frac { h } { 2 } ] ) , } } \\ { { \displaystyle { \bf R } = \left[ \sin \theta \quad - \sin \theta \quad 0 \right] } } \\ { { \displaystyle { \bf R } = \left[ \sin \theta \quad \cos \theta \quad 0 \right] } } \end{array}\tag{7}
$$

with S and R denoting the scaling and rotation matrices, respectively, and $a > 0$ as the scaling hyperparameter.

For a corresponding predicted bounding box $\hat { b } ,$ the same conversion yields a Gaussian distribution $\hat { g } = \mathcal { N } ( \hat { \mu } ; \hat { \Sigma } )$ .

Box Gaussian Loss. Given two Gaussian distributions, their distance is measured as BGL. In this work, the KL divergence is adopted as the distance metric:

$$
\mathcal { L } _ { \mathrm { B G L } } = \frac { 1 } { N _ { b } } \sum _ { i = 1 } ^ { N _ { b } } \mathrm { K L } ( \hat { g } _ { i } , g _ { i } ) ,\tag{8}
$$

where $N _ { b }$ denotes the number of ground-truth bounding boxes;

$$
\mathbf { K L } ( { \hat { g } } , g ) = { \frac { 1 } { 2 } } [ ( { \hat { \mu } } - \mu ) ^ { \top } \Sigma ^ { - 1 } ( { \hat { \mu } } - \mu ) + \operatorname { T r } ( \Sigma ^ { - 1 } { \hat { \Sigma } } ) + \log { \frac { | \Sigma | } { | { \hat { \Sigma } } | } } - 3 ]\tag{9}
$$

represents the KL divergence between the predicted Gaussian distribution $\hat { g }$ and the ground-truth one g. Unlike $\operatorname { K L } ( g , \hat { g } )$ the $\pmb { \Sigma } ^ { - 1 }$ term in the first term of $\operatorname { K L } ( \hat { g } , g )$ remains constant during training, leading to more stable optimization of the Gaussian mean. In addition, to ensure numerical stability during covariance inversion, the predicted bounding box scales are clamped to a small positive lower bound.

The first term $( \hat { \mu } - \mu ) ^ { \top } \Sigma ^ { - 1 } ( \hat { \mu } - \mu )$ is the Mahalanobis distance between $\hat { \mu }$ and $\mathcal { N } ( \boldsymbol { \mu } ; \boldsymbol { \Sigma } )$ . Unlike $L _ { 1 }$ distance, it accounts for the shape and orientation of the ground-truth object, thereby providing more informative localization error. The remaining terms capture differences in scales and orientations.

It should be noted that the hyperparameter a influences only the first term, while its effects in the other terms are canceled through division, making the result independent of a. For larger objects, a should be set higher, since large Euclidean distances may otherwise yield small Mahalanobis distances.

The total regression loss $\mathcal { L } _ { \mathrm { r e g } }$ is defined as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r e g } } = \mathcal { L } _ { \mathrm { o r i \_ r e g } } + \lambda \mathcal { L } _ { \mathrm { B G L } } , } \end{array}\tag{10}
$$

where $\mathcal { L } _ { \mathrm { o r i \_ r e g } }$ denotes the original regression loss used in the detection head $( \mathrm { e } . \mathrm { g } . , L _ { 1 }$ loss in CenterHead [35]), and Î» is the balancing weight. The classification loss remains unchanged.

## IV. EXPERIMENTS

## A. Implementation Details

Datasets and Evaluation Metrics. Two publicly available datasets are used to evaluate the proposed RadarGaussianDet3D: TJ4DRadSet [7] and View-of-Delft (VoD) [8]. Both datasets provide synchronized data from a 4D radar, a camera, and a LiDAR, with tens of thousands of annotated 3D bounding boxes. Compared with VoD, TJ4DRadSet is collected under more complex road and traffic conditions with greater lighting variability, making detection significantly more challenging. Experiments on both TJ4DRadSet and VoD are conducted using their official dataset partitioning, with models trained and evaluated on each dataset independently.

Following the official TJ4DRadSet protocol, both 3D and BEV average precisions (APs) are evaluated over the region $ { \cal D } ~ = ~ [ x _ { \mathrm { m i n } } , x _ { \mathrm { m a x } } ] ~ \times ~ [ y _ { \mathrm { m i n } } , y _ { \mathrm { m a x } } ] ~ \times ~ [ z _ { \mathrm { m i n } } , z _ { \mathrm { m a x } } ] ~ =$ $[ 0 , 6 9 . 1 2 \mathrm { m } ] \times [ - 3 9 . 6 8 \mathrm { m } , 3 9 . 6 8 \mathrm { m } ] \times [ - 4 \mathrm { m } , 2 \mathrm { m } ]$ for car, pedestrian, cyclist, and truck.

For the VoD dataset, the 3D AP is computed for the categories of car, pedestrian, and cyclist in both the entire annotated area $( D _ { \mathrm { E A A } } )$ and the region of interest $( D _ { \mathrm { R O I } } ) \mathrm { : }$

$$
\begin{array} { r l } & { D _ { \mathrm { E A A } } = [ x _ { \mathrm { m i n } } , x _ { \mathrm { m a x } } ] \times [ y _ { \mathrm { m i n } } , y _ { \mathrm { m a x } } ] \times [ z _ { \mathrm { m i n } } , z _ { \mathrm { m a x } } ] } \\ & { \qquad = [ 0 , 5 1 . 2 \mathrm { m } ] \times [ - 2 5 . 6 \mathrm { m } , 2 5 . 6 \mathrm { m } ] \times [ - 3 \mathrm { m } , 2 \mathrm { m } ] , } \\ & { D _ { \mathrm { R O I } } = [ x _ { \mathrm { m i n } } ^ { c } , x _ { \mathrm { m a x } } ^ { c } ] \times [ z _ { \mathrm { m i n } } ^ { c } , z _ { \mathrm { m a x } } ^ { c } ] } \\ & { \qquad = [ - 4 \mathrm { m } , 4 \mathrm { m } ] \times [ 0 \mathrm { m } , 2 5 \mathrm { m } ] , } \end{array}\tag{11}
$$

where ROI is defined in the camera coordinate system, and the coordinates x, y, z carry the superscript c.

Network and Hyper-parameters. The proposed model is developed based on CenterPoint-Pillar [35], with the pillar encoder replaced by the proposed PGE and the BGL incorporated. In BEV Gaussian splatting, the target feature map size (H, W ) is set to (320, 320) for VoD and (432, 496) for TJ4DRadSet, ensuring the same resolution as the output of the pillar encoder with the commonly adopted pillar size of 0.16m [5], [11], [12], for fair comparison. The spherical neighborhood radius in LFA is set to $r = 0 . 3 2 \mathrm { m }$ , where small deviations have only a marginal impact on detection accuracy, and the feature dimension in PGE is fixed to $C = 6 4$

TABLE II
<table><tr><td rowspan="2">Models</td><td rowspan="2">Modality</td><td colspan="4">EAA AP (%)</td><td colspan="4">ROI AP (%)</td><td colspan="4">FPS (Hz)</td></tr><tr><td>Car</td><td>Ped.</td><td>Cyc.</td><td>mAP</td><td>Car</td><td>Ped.</td><td>Cyc.</td><td>mAP</td><td>V100</td><td>3090</td><td>4090</td><td>Xavier</td></tr><tr><td>PointPillars (CVPR&#x27;19) [3]</td><td>R</td><td>38.8</td><td>34.4</td><td>66.9</td><td>46.7</td><td>71.9</td><td>45.1</td><td>88.4</td><td>67.8</td><td>78.4</td><td>178.4</td><td>187.0*</td><td>20.6</td></tr><tr><td>CenterPoint* (CVPR&#x27;21 [35]</td><td>R</td><td>32.7</td><td>38.0</td><td>65.5</td><td>45.4</td><td>62.0</td><td>48.2</td><td>85.0</td><td>65.1</td><td>-</td><td>-</td><td>72.2</td><td>-</td></tr><tr><td>PillarNeXt* (CVPR&#x27;23) [36]</td><td>R</td><td>30.8</td><td>33.1</td><td>62.8</td><td>42.2</td><td>66.7</td><td>39.0</td><td>85.1</td><td>63.6</td><td>-</td><td>-</td><td>67.2</td><td>-</td></tr><tr><td>SMURF (T-IV&#x27;23) [11]</td><td>R</td><td>42.3</td><td>39.1</td><td>71.5</td><td>51.0</td><td>71.7</td><td>50.5</td><td>86.9</td><td>69.7</td><td>30.0</td><td>-</td><td>-</td><td>-</td></tr><tr><td>RadarPillars (ITSC&quot;24) [5]</td><td>R</td><td>41.1</td><td>38.6</td><td>72.6</td><td>50.7</td><td>71.1</td><td>52.3</td><td>87.9</td><td>70.5</td><td>82.8</td><td>179.1</td><td>-</td><td>34.4</td></tr><tr><td>MUFASA (ICANN&#x27;24) [37]</td><td>R</td><td>43.1</td><td>39.0</td><td>68.7</td><td>50.2</td><td>72.5</td><td>50.3</td><td>88.5</td><td>70.4</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>4DRadDet (ICRA&#x27;25) [38]</td><td>R</td><td>42.0</td><td>40.7</td><td>71.6</td><td>51.4</td><td>72.1</td><td>51.2</td><td>88.0</td><td>70.4</td><td>-</td><td>-</td><td>35.2</td><td>-</td></tr><tr><td>MAFF-Net* (RAL&#x27;25) [12]</td><td>R</td><td>42.3</td><td>46.8</td><td>74.7</td><td>54.6</td><td>72.3</td><td>57.8</td><td>87.4</td><td>72.5</td><td>-</td><td>-</td><td>28.7</td><td>-</td></tr><tr><td>RadarGaussianDet3D (Ours)</td><td>R</td><td>40.7</td><td>42.4</td><td>73.0</td><td>52.0</td><td>71.2</td><td>51.7</td><td>89.0</td><td>70.6</td><td>83.2</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SCKD (AAAI&#x27;25) [13]</td><td>R+(L)</td><td>41.9</td><td>43.5</td><td>70.8</td><td>52.1</td><td>77.5</td><td>51.1</td><td>86.9</td><td>71.8</td><td></td><td></td><td>39.3</td><td>-</td></tr></table>

1 In the column of Modality, R denotes radar, and (L) represents that LiDAR is incorporated during training.  
2 In each column, the highest value is in bold, the 2nd highest underlined, and the 3rd highest in italic.  
3 Inference speeds are measured on Tesla V100, RTX 3090, RTX 4090 or AGX Xavier (an embedded device designed for autonomous driving).  
4 The results of models / inference speeds marked with â/â are from [12] / [5], while others from corresponding citations.

<!-- image-->  
Fig. 4. Visualization results of RadarGaussianDet3D on the TJ4DRadSet [7] test set (left) and VoD val set (right). Gray points denote 4D radar points in BEV, orange and blue boxes indicate ground-truth and predicted bounding boxes, respectively, and the red triangle marks the ego-vehicle position.

During training, the scaling factor a in BGL is set to 1 for pedestrians and cyclists, and to 3 for cars and trucks. The loss weight for BGL is fixed to 1.0.

Training Details. The proposed model is implemented using the MMDetection3D framework [39] and trained on a single NVIDIA Tesla V100 GPU for 24 epochs with a batch size of 8. AdamW is employed as the optimizer, with an initial learning rate of $2 \times 1 0 ^ { - 4 }$ scheduled by cosine annealing.

## B. Comparison with State-of-the-Arts

Results on TJ4DRadSet. Table I reports the detection accuracy and inference speed of different models on TJ4DRadSet [7], including the proposed RadarGaussianDet3D. It achieves accuracy comparable to the state-of-the-art MAFF-Net [12] (â0.3% mAP3D, +0.4% mAPBEV), while delivering substantially higher inference speed. This efficiency is maintained even on a lower-performance GPU, leaving room for future integration of more computationally intensive enhancements.

Results on VoD. To further assess generalizability, experiments are conducted on the VoD [8] dataset, with results summarized in Table II. RadarGaussianDet3D ranks second among models with radar single-modality, outperforming most methods and only falling behind the two-stage detector MAFF-

Net [12]. Notably, the performance gap between RadarGaussianDet3D and SCKD [13], which leverages both LiDAR and radar during training, is merely 0.1% mAPEAA, highlighting the effectiveness of the proposed approach.

Inference Speed Considerations. Musiat et al. [5] have benchmarked several models (including PointPillars [3]) across GPUs and embedded devices on the VoD [8] dataset, which enables fairer comparison of inference speeds of different models. From Table II, it can be inferred that models with frame rates significantly below that of PointPillars [3] are unlikely to achieve real-time performance (i.e., FPS > 10Hz) on embedded platforms such as NVIDIA AGX Xavier. Consequently, although MAFF-Net [12] yields the highest accuracy, it is unlikely to meet real-time requirements in practical deployments. In contrast, RadarGaussianDet3D operates even faster than PointPillars [3], suggesting strong potential for real-time deployment on embedded devices in autonomous driving systems. The speed gain of PGE primarily stems from avoiding point padding and redundant computation on empty points in pillar-based encoders, together with the efficiency of Gaussian splatting.

## C. Ablation Study

This section presents ablation studies to evaluate the effectiveness of the proposed components. Results on TJ4DRadSet [7] dataset are reported by default.

Ablations on PGE and BGL. Table III reports the performance of models obtained by gradually adding modules from the CenterPoint-Pillar [35] baseline (a) to the complete Radar-GaussianDet3D (g). Replacing the pillar encoder with Gaussian splatting increases the 3D mAP by 0.1% on TJ4DRadSet, since each point contributes to multiple BEV grids, yielding denser feature maps. Adding LFA/GFA brings additional gains of 1.3% and 1.5% mAP3D, respectively, demonstrating that point-level feature interaction benefits Gaussian attribute prediction. When combined to form the proposed PGE, the performance reaches 33.50% $\mathrm { m A P } _ { 3 \mathrm { D } }$ , indicating that LFA and GFA capture complementary information. Finally, the inclusion of BGL further improves accuracy by 1.6%, attributable to its comprehensive measurement of bounding box differences.

On VoD [8], a similar overall performance trend is observed, except that detection accuracy decreases from (a) to (b) due to the sparser and more irregular radar point distributions. This observation emphasizes the necessity of the proposed LFA and GFA modules, which effectively recover and further improve performance by aggregating local and global point features.

TABLE III  
ABLATION STUDIES OF COMPONENTS ON TJ4DRADSET T E S T SET AND VOD [8] V A L SET
<table><tr><td></td><td colspan="4">Enhancements</td><td colspan="5">TJ4DRadSet 3D AP (%)</td><td colspan="5">TJ4DRadSet BEV AP (%)</td><td colspan="2">VoD mAP (%)</td></tr><tr><td></td><td>GS</td><td>LFA</td><td>GFA</td><td>BGL</td><td>Car</td><td>Ped.</td><td>Cyc.</td><td>Tru.</td><td>mAP</td><td>Car</td><td>Ped.</td><td>Cyc.</td><td>Tru.</td><td>mAP</td><td>EAA</td><td>ROI</td></tr><tr><td>(a)</td><td></td><td></td><td></td><td></td><td>23.01</td><td>23.32</td><td>57.58</td><td>19.25</td><td>30.79</td><td>35.82</td><td>26.25</td><td>62.65</td><td>27.95</td><td>38.42</td><td>46.84</td><td>68.51</td></tr><tr><td>(b)</td><td></td><td></td><td></td><td></td><td>23.05</td><td>23.76</td><td>57.45</td><td>19.30</td><td>30.89</td><td>36.11</td><td>27.63</td><td>61.90</td><td>28.71</td><td>38.59</td><td>46.06</td><td>67.00</td></tr><tr><td>(c)</td><td></td><td>â</td><td></td><td></td><td>22.99</td><td>27.94</td><td>61.50</td><td>16.39</td><td>32.20</td><td>39.53</td><td>30.20</td><td>64.84</td><td>29.45</td><td>41.00</td><td>48.70</td><td>69.12</td></tr><tr><td>(d)</td><td></td><td></td><td>â</td><td></td><td>22.80</td><td>27.04</td><td>61.83</td><td>17.63</td><td>32.33</td><td>34.89</td><td>28.96</td><td>67.22</td><td>26.81</td><td>39.47</td><td>48.94</td><td>69.48</td></tr><tr><td>(e)</td><td>JSSS</td><td>L</td><td>â</td><td></td><td>25.77</td><td>28.08</td><td>64.55</td><td>15.61</td><td>33.50</td><td>39.34</td><td>30.57</td><td>68.30</td><td>26.41</td><td>41.15</td><td>50.18</td><td>69.52</td></tr><tr><td>(f)</td><td></td><td></td><td></td><td></td><td>25.77</td><td>23.47</td><td>58.42</td><td>18.94</td><td>31.65</td><td>42.44</td><td>26.60</td><td>63.04</td><td>28.20</td><td>40.07</td><td>48.30</td><td>69.52</td></tr><tr><td>(g)</td><td>L</td><td>â</td><td>â</td><td>&gt;&gt;</td><td>26.69</td><td>28.18</td><td>65.84</td><td>19.63</td><td>35.08</td><td>41.66</td><td>30.28</td><td>69.62</td><td>26.35</td><td>41.98</td><td>52.04</td><td>70.61</td></tr></table>

<!-- image-->  
Fig. 5. Visualization of BEV feature maps from CenterPoint-Pillar [35] (left) and RadarGaussianDet3D without BGL (right) on the TJ4DRadSet test set. The first row presents initial BEV feature map after the pillar encoder or PGE, the second row displays the BEV feature map output by the backbone and neck, and the last row shows the radar point cloud and final detection results.

Ablations on Predicted Gaussian Attributes in PGE. Table IV presents results for different predicted Gaussian attributes, with all models trained without BGL. Predicting position offsets and opacities leads to performance degradation, confirming that excessive degrees of freedom for Gaussian primitives from sparse radar points are detrimental, likely due to insufficient local context and measurement noise inherent to the radar modality. Optimal performance is obtained when the model focuses on estimating scale and rotation attributes.

BEV Feature Map Visualization. To illustrate the advantages of PGE, BEV feature maps from CenterPoint-Pillar [35] and RadarGaussianDet3D (without BGL) are compared. The only difference between these models lies in the module used to transform radar points into BEV feature maps (pillar encoder vs. PGE). As shown in Fig. 5, PGE produces feature maps with broader activation, capturing continuous road boundaries and clearer foreground objects. After processed by the backbone and neck, the pillar-based feature map becomes noisy and over-activated, a consequence of hallucinations when completing sparse maps. This effect is notably reduced in the PGE output, leading to more accurate detections.

ABLATION STUDIES OF PREDICTED GAUSSIAN ATTRIBUTES  
TABLE IV
<table><tr><td rowspan="2"></td><td colspan="2">Predicted Attributes</td><td rowspan="2">3D AP (%)</td><td rowspan="2">BEV AP (%)</td></tr><tr><td>Position Offset</td><td>Opacity</td></tr><tr><td>(a)</td><td>â</td><td>&gt;&gt;</td><td>32.39</td><td>39.72</td></tr><tr><td>(b)</td><td></td><td></td><td>33.30</td><td>40.14</td></tr><tr><td>(c)</td><td></td><td></td><td>33.50</td><td>41.15</td></tr></table>

TABLE V

THE COMPLEXITY OF DIFFERENT LFA IMPLEMENTATIONS
<table><tr><td>Algorithm</td><td>Avg. Runtime</td><td>Max. Memory</td></tr><tr><td>Traversal</td><td>177.9ms</td><td>202.6MB</td></tr><tr><td>Broadcasting &amp; Masking</td><td>3.9ms</td><td>3981.6MB</td></tr><tr><td>Indexing &amp; Scattering</td><td>0.5ms</td><td>202.6MB</td></tr></table>

Complexity of Different LFA Implementations. Table V compares the runtime and peak memory consumption of the LFA implementations in Fig. 3, evaluated on the TJ4DRadSet [7] test set using an NVIDIA Tesla V100 GPU. As expected, the Traversal method is the slowest. Broadcasting & Masking benefits from PyTorchâs optimized matrix operations and achieves faster runtime, but at the cost of sharply increased memory usage due to repeated point features. In contrast, the proposed Indexing & Scattering algorithm leverages mask sparsity to avoid redundant computation and storage, maintaining low memory consumption while further improving speed.

## V. CONCLUSION

This paper presented RadarGaussianDet3D, a fast and accurate 4D radar-based 3D object detector. To mitigate the inherent sparsity of radar point clouds, the proposed Point Gaussian Encoder (PGE) replaces the conventional pillar encoder by predicting Gaussian primitives from aggregated point features and applying 3D Gaussian Splatting to generate dense BEV feature maps. In addition, the Box Gaussian Loss (BGL) comprehensively measures bounding box differences by accounting for correlations among attributes through boxto-Gaussian conversion. Extensive experiments demonstrated that RadarGaussianDet3D achieves high accuracy while delivering substantially lower inference latency, highlighting its suitability for real-time autonomous driving applications.

By unifying radar points and 3D bounding boxes under Gaussian-based representations, this work contributes new perspectives to 4D radar-based 3D object detection. Future research will extend this paradigm to multi-modal fusion and temporal modeling for perception in complex scenarios.

## REFERENCES

[1] Z. Han, J. Wang, Z. Xu, S. Yang, L. He, S. Xu, and J. Wang, â4D millimeter-wave radar in autonomous driving: A survey,â arXiv preprint arXiv:2306.04242, vol. 1, 2023.

[2] P.-C. Kung, S. Harisha, R. Vasudevan, A. Eid, and K. A. Skinner, âRadarSplat: Radar Gaussian Splatting for High-Fidelity Data Synthesis and 3D Reconstruction of Autonomous Driving Scenes,â arXiv preprint arXiv:2506.01379, 2025.

[3] A. H. Lang, S. Vora, H. Caesar, L. Zhou, J. Yang, and O. Beijbom, âPointPillars: Fast encoders for object detection from point clouds,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 12 697â12 705.

[4] Z. Lin, Z. Liu, Z. Xia, X. Wang, Y. Wang, S. Qi, Y. Dong, N. Dong, L. Zhang, and C. Zhu, âRCBEVDet: Radar-camera fusion in birdâs eye view for 3D object detection,â in Proc. of IEEE/CVF Conf. on Computer Vision and Pattern Recognition, 2024, pp. 14 928â14 937.

[5] A. Musiat, L. Reichardt, M. Schulze, and O. Wasenmuller, âRadarPillars: Â¨ Efficient Object Detection From 4D Radar Point Clouds,â in 2024 IEEE 27th International Conference on Intelligent Transportation Systems (ITSC). IEEE, 2024, pp. 1656â1663.

[6] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3D Gaussian Â¨ splatting for real-time radiance field rendering,â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[7] L. Zheng, Z. Ma, X. Zhu, B. Tan, S. Li, K. Long, W. Sun, S. Chen, L. Zhang, M. Wan et al., âTJ4DRadSet: A 4D radar dataset for autonomous driving,â in 2022 IEEE 25th International Conference on Intelligent Transportation Systems (ITSC). IEEE, 2022, pp. 493â498.

[8] A. Palffy, E. Pool, S. Baratam, J. Kooij, and D. Gavrila, âMulti-class road user detection with 3+ 1D radar in the View-of-Delft dataset,â IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 4961â4968, 2022.

[9] B. Xu, X. Zhang, L. Wang, X. Hu, Z. Li, S. Pan, J. Li, and Y. Deng, âRPFA-Net: A 4D radar pillar feature attention network for 3D object detection,â in 2021 IEEE International Intelligent Transportation Systems Conference (ITSC). IEEE, 2021, pp. 3061â3066.

[10] L. Zheng, S. Li, B. Tan, L. Yang, S. Chen, L. Huang, J. Bai, X. Zhu, and Z. Ma, âRCFusion: Fusing 4-D radar and camera with birdâs-eye view features for 3-D object detection,â IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1â14, 2023.

[11] J. Liu, Q. Zhao, W. Xiong, T. Huang, Q.-L. Han, and B. Zhu, âSMURF: Spatial multi-representation fusion for 3D object detection with 4D imaging radar,â IEEE Transactions on Intelligent Vehicles, vol. 9, no. 1, pp. 799â812, 2024.

[12] X. Bi, C. Weng, P. Tong, B. Fan, and A. Eichberge, âMAFF-Net: Enhancing 3D Object Detection with 4D Radar via Multi-assist Feature Fusion,â IEEE Robotics and Automation Letters, 2025.

[13] R. Xu, Z. Xiang, C. Zhang, H. Zhong, X. Zhao, R. Dang, P. Xu, T. Pu, and E. Liu, âSCKD: Semi-supervised cross-modality knowledge distillation for 4D radar object detection,â in Proc. of the AAAI Conference on Artificial Intelligence, vol. 39, no. 9, 2025, pp. 8933â8941.

[14] Y. Chae, H. Kim, and K.-J. Yoon, âTowards robust 3D object detection with LiDAR and 4D radar fusion in various weather conditions,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 15 162â15 172.

[15] X. Huang, Z. Xu, H. Wu, J. Wang, Q. Xia, Y. Xia, J. Li, K. Gao, C. Wen, and C. Wang, âL4DR: LiDAR-4Dradar fusion for weather-robust 3D object detection,â in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 4, 2025, pp. 3806â3814.

[16] L. Wang, X. Zhang, B. Xv, J. Zhang, R. Fu, X. Wang, L. Zhu, H. Ren, P. Lu, J. Li et al., âInterFusion: Interaction-based 4D radar and LiDAR fusion for 3D object detection,â in 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022, pp. 12 247â12 253.

[17] W. Xiong, J. Liu, T. Huang, Q.-L. Han, Y. Xia, and B. Zhu, âLXL: LiDAR excluded lean 3D object detection with 4D imaging radar and camera fusion,â IEEE Transactions on Intelligent Vehicles, vol. 9, no. 1, pp. 79â92, 2024.

[18] W. Xiong, Z. Zou, Q. Zhao, F. He, and B. Zhu, âLXLv2: Enhanced LiDAR excluded lean 3D object detection with fusion of 4D radar and camera,â IEEE Robotics and Automation Letters, 2025.

[19] X. Bai, Z. Yu, L. Zheng, X. Zhang, Z. Zhou, X. Zhang, F. Wang, J. Bai, and H.-L. Shen, âSGDet3D: Semantics and geometry fusion for 3D object detection using 4D radar and camera,â IEEE Robotics and Automation Letters, 2024.

[20] L. Zheng, J. Liu, R. Guan, L. Yang, S. Lu, Y. Li, X. Bai, J. Bai, Z. Ma, H.-L. Shen et al., âDoracamom: Joint 3D detection and occupancy prediction with multi-view 4D radars and cameras for omnidirectional perception,â arXiv preprint arXiv:2501.15394, 2025.

[21] G. Hess, C. Lindstrom, M. Fatemi, C. Petersson, and L. Svensson, Â¨ âSplatAD: Real-time LiDAR and camera rendering with 3D gaussian splatting for autonomous driving,â in Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition, 2025, pp. 11 982â11 992.

[22] X. Zhou, Z. Lin, X. Shan, Y. Wang, D. Sun, and M.-H. Yang, âDrivingGaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes,â in Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition, 2024, pp. 21 634â21 643.

[23] F. Chabot, N. Granger, and G. Lapouge, âGaussianBeV: 3D gaussian representation meets perception models for BeV segmentation,â in 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2025, pp. 2250â2259.

[24] S.-W. Lu, Y.-H. Tsai, and Y.-T. Chen, âToward real-world bev perception: Depth uncertainty estimation via gaussian splatting,â in Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition, 2025, pp. 17 124â17 133.

[25] X. Bai, C. Zhou, L. Zheng, S.-Y. Cao, J. Liu, X. Zhang, Z. Zhang, and H.-l. Shen, âRaGS: Unleashing 3D Gaussian Splatting from 4D Radar and Monocular Cues for 3D Object Detection,â arXiv preprint arXiv:2507.19856, 2025.

[26] Y. Huang, W. Zheng, Y. Zhang, J. Zhou, and J. Lu, âGaussianFormer: Scene as gaussians for vision-based 3D semantic occupancy prediction,â in European Conf. on Computer Vision. Springer, 2024, pp. 376â393.

[27] Y. Huang, A. Thammatadatrakoon, W. Zheng, Y. Zhang, D. Du, and J. Lu, âGaussianformer-2: Probabilistic gaussian superposition for efficient 3d occupancy prediction,â in Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition, 2025, pp. 27 477â27 486.

[28] S. Zuo, W. Zheng, Y. Huang, J. Zhou, and J. Lu, âGaussianWorld: Gaussian world model for streaming 3D occupancy prediction,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 6772â6781.

[29] S. Xu, F. Li, S. Jiang, Z. Song, L. Liu, and Z.-x. Yang, âGaussianPretrain: A simple unified 3D gaussian representation for visual pre-training in autonomous driving,â arXiv preprint arXiv:2411.12452, 2024.

[30] S. Liu, Q. Liang, Z. Li, B. Li, and K. Huang, âGaussianFusion: Gaussian-Based Multi-Sensor Fusion for End-to-End Autonomous Driving,â arXiv preprint arXiv:2506.00034, 2025.

[31] W. Zheng, J. Wu, Y. Zheng, S. Zuo, Z. Xie, L. Yang, Y. Pan, Z. Hao, P. Jia, X. Lang et al., âGaussianAD: Gaussian-centric end-to-end autonomous driving,â arXiv preprint arXiv:2412.10371, 2024.

[32] C. Qi, H. Su, K. Mo, and L. Guibas, âPointNet: Deep learning on point sets for 3D classification and segmentation,â in Proc. of IEEE Conf. on Computer Vision and Pattern Recognition, 2017, pp. 652â660.

[33] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Å. Kaiser, and I. Polosukhin, âAttention is All you Need,â Advances in neural information processing systems, vol. 30, 2017.

[34] F. Zhang, H. Yang, Z. Zhang, Z. Huang, and Y. Luo, âTT-Occ: Test-Time Compute for Self-Supervised Occupancy via Spatio-Temporal Gaussian Splatting,â arXiv preprint arXiv:2503.08485, 2025.

[35] T. Yin, X. Zhou, and P. Krahenbuhl, âCenter-based 3D object detection and tracking,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 11 784â11 793.

[36] J. Li, C. Luo, and X. Yang, âPillarNeXt: Rethinking network designs for 3D object detection in LiDAR point clouds,â in Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition, 2023, pp. 17 567â17 576.

[37] X. Peng, M. Tang, H. Sun, K. Bierzynski, L. Servadei, and R. Wille, âMUFASA: Multi-view Fusion and Adaptation Network with Spatial Awareness for Radar Object Detection,â in International Conference on Artificial Neural Networks. Springer, 2024, pp. 168â184.

[38] C. Weng, X. Bi, P. Tong, and A. Eichberger, â4DRadDet: Clusterqueried enhanced 3D object detection with 4D radar,â in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 16 984â16 990.

[39] M. Contributors, âMMDetection3D: OpenMMLab next-generation platform for general 3D object detection,â https://github.com/open-mmlab/ mmdetection3d, 2020.