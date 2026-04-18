# SPAGS: Sparse-View Articulated Object Reconstruction from Single State via Planar Gaussian Splatting

Di Wu1,2,4 , Liu Liu 3\*, Xueyu Yuan3, Qiaojun Yu4,5, Wenxiao Chen3, Ruilong Yan1,2,

Yiming Tang3, Liangtu Song1

1 Hefei Institutes of Physical Science, Chinese Academy of Sciences

2 University of Science and Technology of China

3 Hefei University of Technology

4 Shanghai AI Laboratory, China

5 Shanghai Jiao Tong University

Email: wdcs@mail.ustc.edu.cn, liuliu@hfut.edu.cn

## Abstract

Articulated objects are ubiquitous in daily environments, and their 3D reconstruction holds great significance across various fields. However, existing articulated object reconstruction methods typically require costly inputs such as multi-stage and multi-view observations. To address the limitations, we propose a category-agnostic articulated object reconstruction framework via planar Gaussian Splatting, which only uses sparse-view RGB images from a single state. Specifically, we first introduce a Gaussian information field to perceive the optimal sparse viewpoints from candidate camera poses. Then we compress 3D Gaussians into planar Gaussians to facilitate accurate estimation of normal and depth. The planar Gaussians are optimized in a coarse-to-fine manner through depth smooth regularization and few-shot diffusion. Moreover, we introduce a part segmentation probability for each Gaussian primitive and update them by back-projecting part segmentation masks of renderings. Extensive experimental results demonstrate that our method achieves higher-fidelity part-level surface reconstruction on both synthetic and real-world data than existing methods. Codes will be made publicly available.

## 1. Introduction

Articulated objects are prevalent in our daily life, such as drawers and scissors. Reconstructing articulated objects holds significant value across various fields, including embodied intelligence, virtual reality, and robotics. However, high-fidelity reconstruction of articulated objects is a nontrivial task since they vary greatly in size and require partlevel mesh extraction for manipulation.

Under this circumstance, some methods [7, 19] learn category-specific prior knowledge for articulated object reconstruction. Nevertheless, these models struggle with unseen object types. Recently, PARIS [11], REArtGS [25], and ArtGS [13] achieve category-agnostic part-level reconstruction using multi-view RGB or RGBD images from two states of articulated objects. However, the multi-view observations from two states are costly in certain cases, especially for robot manipulation, which requires robots to perform time-consuming motion and complex pose estimation for image acquisition.

<!-- image-->  
Figure 1. Given an arbitrary articulated object, our method enables autonomous optimal sparse viewpoints perception and produces: (1) surface mesh; (2) textured mesh; (3) novel view synthesis; (4) articulated modeling; (5) unseen state generation.

In recent years, some approaches [21, 36] are dedicated to reducing reliance on multi-view reconstruction. Unfortunately, when the input views become extremely sparse (e.g., 4 views), their reconstruction quality suffers a sharp decline. Most recently, GaussianObject [31] incorporates diffusion model [23] with 3D Gaussian Splatting (3DGS) [8], achieving realistic object reconstruction only with 4-view RGB images. Nevertheless, we make observation that GaussianObject still exhibits followings limitations for articulated object reconstruction: (1) The disorderly and irregular nature of 3D Gaussian primitives makes depth and normal estimation difficult, leading to inaccurate surface reconstruction. (2) It requires a predefined view setting, and there is a deviation between the manually selected viewpoints and the optimal observation, affecting its application in robot autonomous perception. (3) It fails to extract partlevel meshes, limiting downstream manipulation of articulated objects.

To tackle the above-mentioned challenges, we propose SPAGS (shown in Fig. 1), the first category-agnostic framework of single-state SParse-view Articulated object reconstruction via Gaussian Splatting, to our best knowledge. Specifically, we first propose an optimal view perception method by establishing a Gaussian information field, which continuously estimates optimal viewpoints with the maximum information potential. We later acquire a structured initialization for Gaussian primitives via a 3D generative model [6] and register the initialization using a pyramid network.

Afterwards, we compress 3D Gaussians into planar Gaussians to facilitate depth and normal estimation. We adopt a coarse-to-fine strategy to optimize the planar Gaussians. In coarse training, we leverage the pseudo depth labels from a depth estimation model and depth smoothness regularization to enhance geometric learning. During refinement, we fine-tune a pretrained diffusion model using few-shot data by constructing image pairs within reliable regions, and then use the fine-tuned model to refine noisy regions. Subsequently, we employ a Visual Language Model (VLM) to generate an articulation tree, and assign part-aware probability to each Gaussian primitive, which is updated via back-projecting 2D part masks. Extensive experiments demonstrate our method significantly outperforms the existing state-of-the-art (SOTA) methods on both synthetic and real-world data.

In summary, our main contributions can be summarized as follows: (1) We propose the first category-agnostic framework of high-fidelity articulated object reconstruction, using only sparse-view RGB images from a single state. (2) We propose the Gaussian information field to estimate the information potential of a sampled camera pose, achieving optimal view perception. (3) We propose a coarse-to-fine optimization strategy for planar Gaussians, improving geometry learning through depth smooth regularization and few-shot diffusion.

## 2. Related Works

## 2.1. Sparse-View Image Reconstruction

While NeRF [17] and 3DGS [8] emerge as novel fashions for high-fidelity scene reconstruction, these methods typically struggle with sparse-view input setting. Therefore, some works attempt to reduce the number of training views. Specifically, RegNeRF [20] proposes additional geometric regularization from unobserved viewpoints. SparseNeRF [24] distills local depth ranking priors from a monocular depth estimation model. Although achieving improved novel view synthesis results, they yield entangled representation from the implicit radiation fields, making it challenging to extract an acceptable surface mesh. Recently, FSGS [36] introduces a proximity-guided Gaussian unpooling to increase the density of 3D Gaussians with sparseview training images. CoherentGS [21] proposes a coherent regularization to optimize structured 3D Gaussians. Sparse2DGS [26] combines MVS with 3DGS for improved sparse-view reconstruction. SparseGS [30] and GaussianObject [31] integrate the diffusion model with 3DGS for sparse 360-degree reconstruction. However, these methods still rely on pre-defined sparse-view selection, and the irregular nature of 3D Gaussian primitives hinders accurate normal and depth estimation.

## 2.2. Articulated Object Surface Reconstruction

Articulated object surface reconstruction has gradually attracted more attention in recent years. ASDF [19] and Ditto [7] learn pre-trained models for shape generation of articulated objects, but they struggle with unseen object types. SINGAPO [12] and DreamArt [14] propose single-view surface generation methods for articulated objects. However, these methods usually yield unfaithful results in unseen regions. Recently, PARIS [11], ArticulatedGS [3], REArtGS [25] and ArtGS [13] achieves categoryagnostic high-fidelity reconstruction of articulated objects with two-stage multi-view RGB images. However, these methods still require costly inputs, limiting their practicality in downstream tasks.

## 3. Methodology

As shown in Fig. 2, given an unseen articulated object with q parts $\left( q \geq 2 \right)$ , our method first perceives optimal sparse viewpoints from candidate camera poses that are randomly sampled, and then achieve part-level surface reconstruction. We provide a detailed elaboration of our pipeline below.

## 3.1. Optimal Viewpoint Perception and Representation Initialization

As illustrated in Algorithm 1, we randomly sample candidate camera poses $\mathcal { V } = \{ \sigma _ { 1 } , . . . , \sigma _ { N } \}$ from the upper hemisphere of an arbitrary articulated object, and only select K sparse views $( K \ll N )$ that maximize visual information gain for sparse 3DGS reconstruction. The initial viewpoint set Ï and 3D Gaussian primitives are initialized with a random viewpoint and corresponding captured image respectively. Generally, for the viewpoint set $\pi ~ = ~ \{ 1 , . . . , k \}$ , we introduce a reliability parameter P for each 3D Gaussian primitive $\mathcal { G } _ { i }$ , and define the Gaussian Information Field (GIF) as following:

<!-- image-->  
Figure 2. The Framework of SPAGS. We use the snowflake symbol to denote frozen network weights and the flame symbol to indicate trainable weights. âReg.â and âRegist.â denote regularization and registration respectively. We highlight our main contributions in green. Our method SPAGS can autonomously perceive the optimal sparse viewpoints and achieve high-fidelity reconstruction results for arbitrary articulated objects.

$$
\Psi _ { i } = - \log P ( 1 : k ) \cdot \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } \left( 1 - \alpha _ { j } \right) ,\tag{1}
$$

where $\alpha$ is the Î±-blending weight and $P ( 1 \ : \ k ) \ \in$ [0, 1] is current reliability, quantifying rendering quality via LPIPS [35] and NIQE [18], without ground truth image. Here, we extend LPIPS result to a pixel-wise heatmap obtained by upsampling the spatial error map. We define the information potential E as: $E = - \log P ( 1 : k ) $ Intuitively, Î¨ encodes Gaussian reliable probability, forming a ray-modulated energy field.

For a candidate view $\sigma ,$ , we define its Information Field Intensity (IFI) as $\scriptstyle { \mathcal { T } } ( \sigma )$ , and is computed via path integration along camera rays r:

$$
\mathcal { T } ( \sigma ) = \oint _ { r ( \sigma ) } \Psi \mathrm { d } r = \sum _ { j = 1 } ^ { n _ { \sigma } } \sum _ { i = 1 } ^ { N _ { \mathcal { G } } } \Psi _ { i } ,\tag{2}
$$

where n means all measurement beams, and $N _ { \mathcal { G } }$ denotes the number of the ordered Gaussians along the ray. We incorporate the $k + 1$ optimal viewpoint $\sigma ^ { * }$ into existing viewpoint set $\pi ,$ and utilize the captured images from updated Ï to optimize the Gaussian primitives by vanilla 3DGS pipeline [8]. Besides, we update P through the accumulated information potential $E$ to ensure $P$ within the defined bounds [0, 1]. The accumulated information potential

Algorithm 1 Optimal View Perception Algorithm   
1: Initialize G, Ï with a random view   
2: while k < K do   
3: while $\sigma \in \mathcal { V } _ { \mathrm { r e m a i n } }$ do   
4: $\mathcal { I } ( \sigma )  \mathcal { G } , \sigma$   
5: end while   
6: Ïâ â argmaxÏ (I(Ï))   
7: $\pi  \pi \cup \{ \sigma ^ { * } \}$   
8: Optimize G with Ï   
9: Update P , Vremain   
10: end while

$E ( 1 : k + 1 )$ is formulated as:

$$
E ( 1 : k + 1 ) = E ( 1 : k ) + E ( k + 1 )\tag{3}
$$

and $P ( 1 : k + 1 )$ is updated by:

$$
P ( 1 : k + 1 ) = \exp \left( - E \left( 1 : k + 1 \right) \right)\tag{4}
$$

Note that GauSS-MI [29] proposes a method similar to ours, but GauSS-MI requires the observations or expected RGB results corresponding to the candidate viewpoints. In contrast, our method conducts optimal viewpoint perception only with candidate camera poses from random sampling, which is more convenient for downstream applications.

Once we finish sparse view perception, we attempt to obtain an initial initialization $\mathbf { p _ { \lambda } \in \mathbb { R } ^ { 3 } }$ using a monocular 3D generation model SPAR3D [6]. However, this is not a ready-to-use pipeline since the generated pose and scale are inaccurate, which seriously affects subsequent reconstruction. We observe that although the points $\bar { \mathbf { p } } \in \mathbb { R } ^ { 3 }$ of Gaussian primitives from view perception are noisy, the global pose and scale of Â¯p are relatively accurate. Therefore, we seek to distill a structured representation with reasonable pose and scale through registering p and Â¯p.

Due to the complex disparities in pose and scale between p and Â¯p, we assume that each individual point $\mathbf { p } _ { i } \in \mathbf { p }$ undergoes a similarity transformation to $\hat { \mathbf { p } } _ { i }$ . We employ an exponential mapping $\mathbf { T } = ( \omega , \mathbf { t } , s )$ to parameterize the similarity transformation, where s is the scaling factor and t is the translation. $\omega \in \mathbb { R } ^ { 3 }$ is an axis-angle vector. Inspired by [10], we adopt a pyramid structure network to progressively learn the deformation parameters T which is decomposed into a sequence $\{ \mathbf { T } _ { 1 } , \mathbf { T } _ { 2 } , . . . , \mathbf { T } _ { l } \}$ , with $l = 9$ . Please refer to the supplementary material for detailed elaboration.

## 3.2. Coarse Training for Planar 3D Gaussians

After obtaining the structured initialization p for Gaussian primitives, we first compress the 3D Gaissians to planar Gaussians, facilitating more reasonable depth and normal estimation. Concretely, we introduce a scale loss $\mathcal { L } _ { \mathrm { s c a l e } }$ to flatten 3D Gaussian primitives into 2D planes.

$$
\mathcal { L } _ { \mathrm { s c a l e } } = \frac { 1 } { N _ { g } } \sum _ { i } ^ { N _ { g } } \| \operatorname* { m i n } ( S _ { 1 } , S _ { 2 } , S _ { 3 } ) \|\tag{5}
$$

where $S _ { 1 } , S _ { 2 } , S _ { 3 }$ denote the scale of Gaussian primitive $\mathcal { G } _ { i }$ along each direction. Based on the planar Gaussian primitives, we can easily acquire the normal via the shortest axis and viewing direction. Denoting the normal rendering from Î±-blending as $N _ { \ast }$ , we can derive the unbiased depth D following PGSR [1], which are illustrated in the supplementary materials.

To enhance geometric constraints for the planar Gaussians, we employ a monocular depth estimation model [32] to introduce additional depth cues D for sparse-view images. Considering that the depth estimation result D is typically noisy, we employ scale parameter $\varphi =$ $\left\{ \varphi _ { 1 } , \varphi _ { 2 } , . . . , \varphi _ { K } \right\}$ and offset parameter $\pmb { \eta } = \left\{ \eta _ { 1 } , \eta _ { 2 } , . . . , \eta _ { K } \right\}$ to adaptively refine the estimated depth $\mathcal { D } _ { i }$ for each view. Specifically, we set $\varphi$ as a learnable parameter to align depth scale. We optimize the parameters of an image convolutional decoder to update the offset Î· of whole image instead of estimating pixel-wise offset, ensuring a coherent variation. Given the depth rendering D of planar Gaussians, the depth regularization ${ \mathcal { L } } _ { \mathrm { d e p t h } }$ can be formulated as:

$$
\mathcal { L } _ { \mathrm { d e p t h } } = \sum _ { i } ^ { K } \| \mathbf { D } _ { i } - ( a _ { i } \mathcal { D } _ { i } + \pmb { \eta } _ { i } ) \| _ { 1 }\tag{6}
$$

Moreover, we encourage the depth variation of Gaussian primitives to be smooth. We start with adopting the c-channel segmentation module from [21] to segment each estimated depth $\mathcal { D } _ { i }$ into c similar regions, as shown in part (3) of Fig. 2. For each region, we introduce a local depth constraint ${ \mathcal { L } } _ { \mathrm { s m o o t h } }$ to ensure the depth variations are smooth, which are expressed as:

$$
\mathcal { L } _ { \mathrm { s m o o t h } } = \sum _ { i } ^ { K } \mathcal { E } ( x _ { i } ) \odot \sum _ { j } ^ { c } \left( \left| \nabla _ { x } \mathbf { D } _ { i } ^ { j } \right| + \left| \nabla _ { y } \mathbf { D } _ { i } ^ { j } \right| \right)\tag{7}
$$

where $\mathcal { E }$ is an edge detection model that returns binary edge mask for image $x _ { i } ,$ using Sobel operator to eliminate the impact of depth smooth regularization on image edges.

The coarse training objective can be formulated as: ${ \mathcal { L } } _ { \mathrm { c o a r s e } } = { \mathcal { L } } _ { \mathrm { c o l o r } } + { \mathcal { L } } _ { \mathrm { p l a n a r } }$ , where $\scriptstyle { \mathcal { L } } _ { \mathrm { c o l o r } }$ is the color loss of pixel space using in $[ 8 ] . \mathcal { L } _ { \mathrm { p l a n a r } }$ is expressed as:

$$
{ \mathcal { L } } _ { \mathrm { p l a n a r } } = \lambda _ { \mathrm { s c a l e } } { \mathcal { L } } _ { \mathrm { s c a l e } } + \lambda _ { \mathrm { d e p t h } } { \mathcal { L } } _ { \mathrm { d e p t h } } + \lambda _ { \mathrm { s m o o t h } } { \mathcal { L } } _ { \mathrm { s m o o t h } }\tag{8}
$$

## 3.3. Planar Gaussian Refinement via Few-shot Diffusion

To optimize the representation of regions lacking supervision, we introduce the diffusion model [22] to futher refine Gaussian primitives. Note that our approach only utilize few-shot data to fine-tune the diffusion model, i.e., the sparse-view inputs.

Concretely, we first establish reliable regions â¦ and noisy regions $\Omega ^ { \mathrm { n o i s y } }$ . For each available viewpoint $\sigma _ { i } .$ we define the reliable regions $\Omega _ { i }$ as:

$$
\Omega _ { i } = \left\{ { \bf R } _ { i } ^ { \mathrm { g e n } } , { \bf t } _ { i } ^ { \mathrm { g e n } } | \Delta \theta _ { z } , \Delta \theta _ { y } , \Delta \theta _ { x } , \Delta { \bf t } | \right\}\tag{9}
$$

where $\mathbf { R } _ { i } ^ { \mathrm { g e n } }$ and $\mathbf { t } _ { i } ^ { \mathrm { g e n } }$ represent reliable generated rotations with a limited range for Euler angle perturbations $\Delta \theta _ { z }$ , $\Delta \theta _ { y } , \Delta \theta _ { x } ,$ and translations with perturbation ât respectively. In turn, we define the camera poses that fall outside this range as the noisy regions. We then randomly sample a generated camera pose $\bar { \sigma } _ { i }$ from each reliable region, and take its rendering xÂ¯ as pseudo label for the diffusion model.

To augment the training data, we employ the leaveone-out strategy to generate additional image pairs, similar as [31]. We divide the input data into n subsets, each of which contains $n - 1$ original images and one left-out image $x _ { \mathrm { l e f t } }$ . For each subset, we train the planar Gaussians for 6,000 iterations without $x _ { \mathrm { l e f t } }$ to yield the degraded renderings $x _ { \mathrm { d e g } }$ and $\bar { x } _ { \mathrm { d e g } }$ for $x _ { \mathrm { l e f t } }$ and its corresponding generated label $\bar { x } _ { \mathrm { l e f t } }$ respectively. Subsequently, we incorporate the left-out image $x _ { \mathrm { l e f t } }$ to continue training the Gaussian primitives $4 { , } 0 0 0$ iterations, producing the repaired rendering for $x _ { \mathrm { d e g } }$ and $\bar { x } _ { \mathrm { d e g } }$ . These renderings at different iterations are combined with $x _ { \mathrm { l e f t } }$ and $\bar { x } _ { \mathrm { l e f t } }$ to form image pairs for diffusion model fine-tuning.

We use a pre-trained ControlNet [34] to steer the diffusion model with conditional images. Specially, we add LoRA [4] layers to the transformer blocks in the diffusion

Table 1. Quantitative results for the surface reconstruction quality on PartNet-Mobility dataset. â denotes methods designed for multi-view reconstruction. We bold the best results and underline the second best results.
<table><tr><td>Metrics</td><td>Method</td><td>Stapler</td><td>USB</td><td>Scissor</td><td>Fridge</td><td>Foldchair</td><td>Washer</td><td>Blade</td><td>Laptop</td><td>Oven</td><td>Storage</td><td>Mean</td></tr><tr><td rowspan="5">CD â</td><td>PGSR* [1]</td><td>13.19</td><td>26.94</td><td>6.01</td><td>672.76</td><td>32.20</td><td>32.40</td><td>2.49</td><td>24.12</td><td>48.68</td><td>81.90</td><td>94.07</td></tr><tr><td>CoherentGS [21]</td><td>324.65</td><td>33.76</td><td>313.04</td><td>40.87</td><td>80.61</td><td>64.31</td><td>9.34</td><td>164.08</td><td>41.42</td><td>303.67</td><td>137.58</td></tr><tr><td>Sparse2DGS [26]</td><td>49.25</td><td>35.57</td><td>28.96</td><td>38.87</td><td>12.64</td><td>425.27</td><td>4.13</td><td>28.47</td><td>73.74</td><td>60.88</td><td>75.78</td></tr><tr><td>GaussianObject [31]</td><td>7.39</td><td>6.92</td><td>0.82</td><td>36.05</td><td>0.58</td><td>68.87</td><td>1.58</td><td>166.03</td><td>12.30</td><td>126.93</td><td>42.75</td></tr><tr><td>SPAGS (Ours)</td><td>5.65</td><td>6.56</td><td>0.89</td><td>9.04</td><td>0.66</td><td>27.54</td><td>1.05</td><td>5.21</td><td>17.56</td><td>21.80</td><td>9.60</td></tr><tr><td rowspan="5">F1â</td><td>PGSR* [1]</td><td>0.08</td><td>0.13</td><td>0.33</td><td>0.02</td><td>0.15</td><td>0.05</td><td>0.25</td><td>0.19</td><td>0.03</td><td>0.01</td><td>0.12</td></tr><tr><td>CoherentGS [21]</td><td>0.01</td><td>0.06</td><td>0.08</td><td>0.06</td><td>0.15</td><td>0.03</td><td>0.32</td><td>0.17</td><td>0.02</td><td>0.00</td><td>0.09</td></tr><tr><td>Sparse2DGS [26]</td><td>0.11</td><td>0.07</td><td>0.24</td><td>0.05</td><td>0.18</td><td>0.03</td><td>0.16</td><td>0.13</td><td>0.03</td><td>0.04</td><td>0.10</td></tr><tr><td>GaussianObject [31]</td><td>0.10</td><td>0.15</td><td>0.38</td><td>0.05</td><td>0.35</td><td>0.05</td><td>0.27</td><td>0.10</td><td>0.03</td><td>0.01</td><td>0.15</td></tr><tr><td>SPAGS (Ours)</td><td>0.14</td><td>0.19</td><td>0.40</td><td>0.06</td><td>0.38</td><td>0.05</td><td>0.32</td><td>0.13</td><td>0.03</td><td>0.02</td><td>0.17</td></tr><tr><td rowspan="5">EMD â</td><td>PGSR* [1]</td><td>0.14</td><td>0.19</td><td>0.07</td><td>0.84</td><td>0.16</td><td>0.17</td><td>0.06</td><td>0.13</td><td>0.22</td><td>0.23</td><td>0.22</td></tr><tr><td>CoherentGS [21]</td><td>0.53</td><td>0.26</td><td>0.44</td><td>0.20</td><td>0.32</td><td>0.23</td><td>0.06</td><td>0.16</td><td>0.25</td><td>0.60</td><td>0.31</td></tr><tr><td>Sparse2DGS [26]</td><td>0.50</td><td>0.59</td><td>0.64</td><td>0.62</td><td>0.30</td><td>0.72</td><td>0.50</td><td>0.51</td><td>0.28</td><td>0.31</td><td>0.50</td></tr><tr><td>GaussianObject [31]</td><td>0.17</td><td>0.12</td><td>0.09</td><td>0.12</td><td>0.09</td><td>0.20</td><td>0.05</td><td>0.31</td><td>0.17</td><td>0.38</td><td>0.17</td></tr><tr><td>SPAGS (Ours)</td><td>0.12</td><td>0.11</td><td>0.09</td><td>0.10</td><td>0.08</td><td>0.17</td><td>0.04</td><td>0.10</td><td>0.16</td><td>0.20</td><td>0.12</td></tr></table>

U-Net and the ControlNet U-Net. The fine-tuning objective ${ \mathcal { L } } _ { \mathrm { d i f f } }$ can be formulated as:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { d i f f } } = \mathbb { E } \left[ \left. \left( \epsilon _ { \theta } \left( x _ { \mathrm { l e f t } } , t , x _ { \mathrm { d e g } } \right) - \epsilon \right) \right. _ { 2 } ^ { 2 } \right] } \\ & { ~ + \lambda _ { \mathrm { g e n } } \mathbb { E } \left[ \left. \left( \epsilon _ { \theta } \left( \bar { x } _ { \mathrm { l e f t } } , t , \bar { x } _ { \mathrm { d e g } } \right) - \epsilon \right) \right. _ { 2 } ^ { 2 } \right] } \end{array}\tag{10}
$$

where Ïµ and $\epsilon _ { \theta }$ represent the added noise and predicted noise respectively, and t is the diffusion step.

Once the diffusion model fine-tuning is completed, we utilize it to optimize the noise regions $\Omega _ { \mathrm { n o i s y } }$ . We start with randomly sampling viewpoints $\sigma _ { \mathrm { r a n d } }$ from $\Omega _ { \mathrm { n o i s y } } .$ , and obtain corresponding rendering $x ( \mathcal { G } , \sigma _ { \mathrm { r a n d } } )$ . Following [16], we use a latent diffusion encoder Îµ to encode the rendering, later employ the image condition to yield a noisy latent representation $\mathbf { z } _ { t } .$

$$
\mathbf { z } _ { t } = \sqrt { \bar { \alpha } _ { t } } \varepsilon \left( x ( \mathcal { G } , \sigma _ { \mathrm { r a n d } } ) \right) + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon\tag{11}
$$

where Î±Â¯ is a predefined coefficient. Then a denoised sample $x _ { \mathrm { g e n } }$ can be derived from $\mathbf { z } _ { t }$ through the diffusion model. Subsequently, we leverage the image pairs $( x _ { \mathrm { g e n } } , x ( \mathcal { G } , \sigma _ { \mathrm { r a n d } } ) )$ to refine the coarse Gaussian primitives ${ \mathcal { G } } ,$ , and the optimization loss ${ \mathcal { L } } _ { \mathrm { r e p a i r } }$ is expressed as:

$$
{ \mathcal { L } } _ { \mathrm { r e p a i r } } = \left\| x _ { \mathrm { g e n } } - x ( { \mathcal { G } } , \sigma _ { \mathrm { r a n d } } ) ) \right\| _ { 2 } + f ( x _ { \mathrm { g e n } } - x ( { \mathcal { G } } , \sigma _ { \mathrm { r a n d } } ) )\tag{12}
$$

where f denotes the perceptual similarity function LPIPS.

Besides, we introduce a view consistency regularization ${ \mathcal { L } } _ { \mathrm { v c } }$ to ensure the corresponding pixels rendered with adjacent viewpoints originate from the same 3D points. Specifically, the 3D positions back-projected from the rendering pixel batch $\delta _ { i }$ under viewpoint $\sigma _ { i }$ should align with the corresponding rendering pixel batch under the generated camera pose $\bar { \sigma } _ { i }$ . Therefore, ${ \mathcal { L } } _ { \mathrm { v c } }$ can be represented as following:

$$
\mathcal { L } _ { \mathrm { v c } } = \frac { 1 } { | \delta | } \sum _ { i } \| ( \Theta ( \delta _ { i } | \sigma _ { i } ) ) - ( \Theta ( H \delta _ { i } | \bar { \sigma } _ { i } ) ) \|\tag{13}
$$

where Î denotes the back-projection operation via the depth rendering, and H is the homography matrix follwing [1], mapping $\delta _ { i }$ to its corresponding pixel batch under $\bar { \sigma } _ { i } .$ . In this way, we enhance the geometric consistency of the planar Gaussians. The total training objective of the refinement is formulated as: ${ \mathcal { L } } _ { \mathrm { r e f i n e } } = { \mathcal { L } } _ { \mathrm { r e p a i r } } + \lambda _ { \mathrm { v c } } { \mathcal { L } } _ { \mathrm { v c } } + { \mathcal { L } } _ { \mathrm { p l a n a r } } .$

## 3.4. Articulation Modeling for Planar Gaussians

After the coarse-to-fine optimization, we employ GPT-4o to generate an abstract articulated tree for planar Gaussians, taking multiple rendered images as prompts. The articulated tree describes the name and connectivity of each part as well as the joint type.

Subsequently, we introduce a part-aware probability $m _ { i }$ for each Gaussian primitive $\mathcal { G } _ { i }$ . We render Gaussian primitives from a series of viewpoints to yield multiple rendered images, then we employ the Lang-SAM [15] to generate 2D part-level segmentation masks M for these renderings, using the part names from the articulated tree as text prompts. Following [2], the probability $m _ { i } ^ { o }$ of $\mathcal { G } _ { i }$ belongs to part o can be derived through back-projecting M of corresponding pixel $\rho \colon$

$$
m _ { i } ^ { o } = \mathcal { M } ^ { o } ( \rho ) \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } )\tag{14}
$$

In accordance with whether $m _ { i } ^ { o }$ exceeds a predefined threshold $\tau ,$ we are able to determine whether the Gaussian belongs to part o. When finishing the part-level segmentation of the planar Gaussians, we employ TSDF Fusion to extract the surface mesh and set voxel size to 0.004.

<!-- image-->  
Figure 3. Illustration of joint estimation. Note that we use highresolution rendering to query GPT-4o in actual inference.

After obtaining the part-level Gaussian primitives, we first determine a connecting area between two neighboring parts through a distance threshold from Gaussian center to its neighboring part. Then we use the renderings of these clusters and the articulated tree as prompts to query VLM for joint parameters, as shown in Fig. 3. Please refer to the supplementary materials for a detailed elaboration.

## 4. Experiments

## 4.1. Experimental Setting

The comparisons include SOTA methods: PGSR [1] (planar Gaussians), ArtGS [13], REArtGS [25] (category-agnostic reconstruction), CoherentGS [21], Sparse2DGS [26], GaussianObject [31] (sparse-view reconstruction). Except for ArtGS and REArtGS, all other methods are trained using the same 4-view RGB images as ours, while ArtGS and REArtGS are provided with additional 4-view images from another motion state.

We conduct all experiments on a single RTX 4090 GPU. Please refer to the supplementary materials for detailed implementation and metrics calculation.

Datasets. We conduct our experiments on both synthesis and real-world data. For synthesis data, we select the same 10 categories from PartNet-Mobility [27] as REArtGS. Each object contains 64 randomly sampled candidate camera poses and corresponding RGB images. For real-world data, we scan 6 categories of articulated objects, characterized by a diverse scale and geometry. Each object contains 76 to 160 randomly sampled candidate camera poses and corresponding RGB images captured in unbounded indoor scenes. We apply the optimal view selection algorithm to determine 4 optimal sparse views for each object in the two datasets, and use SAM [9] to generate object masks in real-world data. Note that we only use the candidate camera poses without images in view perception.

## 4.2. Comparison with State-of-the-Art Methods

Mesh Reconstruction. We use the Chamfer distance (CD), F1-score, and EMD [33] as the metrics for surface mesh quality. We provide the quantitative results in Table. 1 and present qualitative results in Fig. 4. More qualitative results can be found in the supplementary materials. Our method achieve bests mean results across all metrics with 9.60, 0.17 and 0.12, outperforming other methods significantly. We observe that Sparse2DGS exhibits lots of artifacts under sparse view supervisions, as illustrated in Fig. 4. GaussianObject and CoherentGS produce surface meshes with lots of artifacts, primarily caused by the lack of geometric constraints. In contrast, our method maintains high-fidelity reconstruction by the coarse-to-fine optimization of planar Gaussians.

<!-- image-->  
Figure 4. The qualitative results of whole mesh reconstruction on PartNet-Mobility dataset.

<!-- image-->  
Figure 5. The qualitative results of novel view synthesis on PartNet-Mobility dataset.

Novel View Synthesis Performance. We employ PSNR, SSIM, and LPIPS as metrics for novel view synthesis. Table. 2 shows the quantitative results, and Fig. 5 presents qualitative comparisons. More qualitative results can be found in the supplementary materials. Our method achieves best results across all metrics with 24.13, 0.94 and 0.07. As shown in Fig. 5, our method exhibits more realistic rendering results, while other methods produce significant artifacts and noise in renderings, especially for Sparse2DGS and CoherentGS. This is mainly because our method enhances noise region reconstruction through few-shot diffusion.

Table 2. Quantitative results for the novel view synthesis on PartNet-Mobility dataset. â denotes methods designed for multi-view reconstruction. We bold the best results and underline the second best results.
<table><tr><td>Metrics</td><td>Method</td><td>Stapler</td><td>USB</td><td>Scissor</td><td>Fridge</td><td>Foldchair</td><td>Washer</td><td>Blade</td><td>Laptop</td><td>Oven</td><td>Storage</td><td>Mean</td></tr><tr><td rowspan="4">PSNR â</td><td>PGSR* [1]</td><td>19.68</td><td>19.55</td><td>24.75</td><td>11.90</td><td>12.70</td><td>17.60</td><td>27.82</td><td>22.9</td><td>17.36</td><td>17.85</td><td>19.19</td></tr><tr><td>CoherentGS [21]</td><td>11.54</td><td>11.99</td><td>13.60</td><td>12.85</td><td>9.31</td><td>15.12</td><td>16.88</td><td>12.64</td><td>13.63</td><td>7.93</td><td>12.55</td></tr><tr><td>Sparse2DGS [2]</td><td>21.80</td><td>17.51</td><td>25.15</td><td>22.00</td><td>22.29</td><td>19.79</td><td>27.79</td><td>16.07</td><td>19.12</td><td>22.21</td><td>21.37</td></tr><tr><td>GaussianObject [31]</td><td>23.42</td><td>22.17</td><td>25.52</td><td>23.11</td><td>20.94</td><td>19.04</td><td>28.85</td><td>21.46</td><td>20.53</td><td>14.41</td><td>21.95</td></tr><tr><td rowspan="5">SSIM â</td><td>SPAGS (Ours)</td><td>26.63</td><td>24.62</td><td>27.19</td><td>23.38</td><td>23.14</td><td>20.48</td><td>32.46</td><td>22.71</td><td>20.07</td><td>20.63</td><td>24.13</td></tr><tr><td>PGSR* [1]</td><td>0.93</td><td>0.91</td><td>0.95</td><td>0.63</td><td>0.76</td><td>0.89</td><td>0.97</td><td>0.94</td><td>0.84</td><td>0.86</td><td>0.87</td></tr><tr><td>CoherentGS [21]</td><td>0.85</td><td>0.83</td><td>0.92</td><td>0.82</td><td>0.76</td><td>0.86</td><td>0.94</td><td>0.91</td><td>0.77</td><td>0.57</td><td>0.82</td></tr><tr><td>Sparse2DGS [26]</td><td>0.96</td><td>0.90</td><td>0.96</td><td>0.83</td><td>0.89</td><td>0.70</td><td>0.98</td><td>0.91</td><td>0.89</td><td>0.91</td><td>0.89</td></tr><tr><td>GaussianObject [31]</td><td>0.96</td><td>0.92</td><td>0.96</td><td>0.92</td><td>0.91</td><td>0.88</td><td>0.98</td><td>0.91</td><td>0.87</td><td>0.78</td><td>0.90</td></tr><tr><td rowspan="5">LPIPS â</td><td>SPAGS (Ours)</td><td>0.98</td><td>0.96</td><td>0.97</td><td>0.95</td><td>0.92</td><td>0.94</td><td>0.99</td><td>0.94</td><td>0.89</td><td>0.89</td><td>0.94</td></tr><tr><td>PGSR* [1]</td><td>0.07</td><td>0.10</td><td>0.06</td><td>0.43</td><td>0.22</td><td>0.13</td><td>0.02</td><td>0.07</td><td>0.19</td><td>0.22</td><td>0.15</td></tr><tr><td>CoherentGS [21]</td><td>0.17</td><td>0.19</td><td>0.10</td><td>0.26</td><td>0.25</td><td>0.20</td><td>0.09</td><td>0.09</td><td>0.27</td><td>0.53</td><td>0.22</td></tr><tr><td>Sparse2DGS [2]</td><td>0.06</td><td>0.09</td><td>0.12</td><td>0.12</td><td>0.10</td><td>0.11</td><td>0.03</td><td>0.14</td><td>0.12</td><td>0.15</td><td>0.10</td></tr><tr><td>GaussianObject [31]</td><td>0.04</td><td>0.12</td><td>0.03</td><td>0.09</td><td>0.07</td><td>0.15</td><td>0.02</td><td>0.09</td><td>0.14</td><td>0.29</td><td>0.10</td></tr><tr><td></td><td>SPAGS (Ours)</td><td>0.03</td><td>0.05</td><td>0.03</td><td>0.07</td><td>0.07</td><td>0.10</td><td>0.01</td><td>0.07</td><td>0.13</td><td>0.18</td><td>0.07</td></tr></table>

Table 3. Quantitative results for part-level mesh reconstruction on PartNet-Mobility dataset. CD-d and CD-s represent the CD results for dynamic parts and static parts respectively.
<table><tr><td>Metrics</td><td>Method</td><td>Stapler</td><td>USB</td><td>Scissor</td><td>Fridge</td><td>Foldchair</td><td>Washer</td><td>Blade</td><td>Laptop</td><td>Oven</td><td>Storage</td><td>Mean</td></tr><tr><td rowspan="3">CD-d â</td><td>REArtGS [25]</td><td>98.57</td><td>56.28</td><td>32.41</td><td>128.09</td><td>204.30</td><td>473.19</td><td>131.76</td><td>63.37</td><td>489.21</td><td>171.98</td><td>184.92</td></tr><tr><td>ArtGS [13]</td><td>137.51</td><td>28.47</td><td>49.75</td><td>158.24</td><td>134.29</td><td>507.07</td><td>85.84</td><td>104.01</td><td>543.75</td><td>178.21</td><td>192.71</td></tr><tr><td>SPAGS (Ours)</td><td>27.74</td><td>6.00</td><td>6.26</td><td>23.94</td><td>0.95</td><td>5.69</td><td>17.85</td><td>8.44</td><td>95.14</td><td>66.72</td><td>25.87</td></tr><tr><td rowspan="3">CD-s â</td><td>REArtGS [25]</td><td>89.51</td><td>52.20</td><td>107.36</td><td>85.47</td><td>103.98</td><td>184.71</td><td>26.44</td><td>218.51</td><td>115.60</td><td>36.69</td><td>102.05</td></tr><tr><td>ArtGS [13]</td><td>62.65</td><td>64.01</td><td>77.51</td><td>99.61</td><td>195.07</td><td>271.24</td><td>16.27</td><td>192.04</td><td>100.34</td><td>25.10</td><td>110.38</td></tr><tr><td>SPAGS (Ours)</td><td>66.82</td><td>42.01</td><td>8.80</td><td>48.63</td><td>23.44</td><td>18.25</td><td>1.80</td><td>21.75</td><td>53.74</td><td>24.92</td><td>31.02</td></tr></table>

Table 4. Quantitative results for mesh reconstruction on real-world data.
<table><tr><td>Metrics</td><td>Method</td><td>Stapler</td><td>USB</td><td>Scissor</td><td>Knife</td><td>Drawer</td><td>Bin</td><td>Mean</td></tr><tr><td rowspan="4">CD â</td><td>CoherentGS [21]</td><td>30.65</td><td>10.92</td><td>13.12</td><td>8.76</td><td>104.02</td><td>64.84</td><td>38.72</td></tr><tr><td>Sparse2DGS [26]</td><td>34.07</td><td>7.63</td><td>15.24</td><td>3.21</td><td>84.15</td><td>65.38</td><td>34.95</td></tr><tr><td>GaussianObject [31]</td><td>11.24</td><td>3.09</td><td>5.23</td><td>4.76</td><td>21.54</td><td>15.30</td><td>10.19</td></tr><tr><td>SPAGS (Ours)</td><td>3.13</td><td>2.34</td><td>5.76</td><td>2.82</td><td>16.36</td><td>4.00</td><td>5.74</td></tr><tr><td rowspan="4">F1â</td><td>CoherentGS [21]</td><td>0.11</td><td>0.14</td><td>0.02</td><td>0.16</td><td>0.01</td><td>0.02</td><td>0.08</td></tr><tr><td>Sparse2DGS [26]</td><td>0.14</td><td>0.20</td><td>0.02</td><td>0.21</td><td>0.03</td><td>0.03</td><td>0.11</td></tr><tr><td>GaussianObject [31]</td><td>0.16</td><td>0.25</td><td>0.05</td><td>0.22</td><td>0.01</td><td>0.07</td><td>0.13</td></tr><tr><td>SPAGS (Ours)</td><td>0.18</td><td>0.24</td><td>0.10</td><td>0.27</td><td>0.03</td><td>0.07</td><td>0.15</td></tr><tr><td rowspan="4">EMD â</td><td>CoherentGS [21]</td><td>0.25</td><td>0.23</td><td>0.35</td><td>0.15</td><td>0.64</td><td>0.59</td><td>0.37</td></tr><tr><td>Sparse2DGS [26]</td><td>0.16</td><td>0.18</td><td>0.23</td><td>0.18</td><td>0.58</td><td>0.50</td><td>0.31</td></tr><tr><td>GaussianObject [31]</td><td>0.09</td><td>0.11</td><td>0.16</td><td>0.14</td><td>0.23</td><td>0.27</td><td>0.17</td></tr><tr><td>SPAGS (Ours)</td><td>0.06</td><td>0.07</td><td>0.10</td><td>0.07</td><td>0.13</td><td>0.08</td><td>0.09</td></tr></table>

Articulated Modeling. We denote the CD results for dynamic and static part meshes as CD-d and CD-s respectively. The quantitative results of part-level mesh reconstruction are presented in Table. 3 and the qualitative results are shown in Fig. 6. We also provide quantitative results of joint estimation in the supplementary materials. Although ArtGS and REArtGS utilize two-stage observations, our method still achieve bests mean results on both CD-d and CD-s with 25.87, 31.02, significantly exceeding them. As shown in Fig. 6, our method yields highfidelity part-level mesh reconstruction and accurate joint estimation. This demonstrates our part segmentation and joint estimation for planar Gaussians are simple but effective, achieving realistic results without two-stage observations.

<!-- image-->  
Figure 6. The qualitative results of articulated modeling. We set the dynamic parts to yellow and the static parts to blue, and use red arrows to represent the joints.

Table 5. Ablation of view perception on PartNet-Mobility dataset. Mean results are reported.
<table><tr><td>Settings</td><td>CD â CD-d â</td><td>CD-s â</td></tr><tr><td rowspan="2">Random Sampling Pre-defined</td><td>14.89 40.67</td><td>78.01</td></tr><tr><td>10.23 32.34</td><td>53.02</td></tr><tr><td>Optimal Perception</td><td>9.60 27.87</td><td>44.91</td></tr></table>

## 4.3. Ablation Study

Ablation on View Perception. We conduct the ablation study of view perception on PartNet-Mobility dataset. Quantitative results are shown in Table. 5. âRandom Samplingâ denotes selecting 4-view images randomly from candidates and we take the average results over 10 trials. âPredefinedâ refers to manually choosing promising 4-view images covering 360 degrees around the objects.

Using optimal view perception achieves the best whole surface and part-level reconstruction performance across all metrics, indicating that the Gaussian information fields capture the maximum information potential gain.

Table 6. Ablation of the number of input views on PartNet-Mobility dataset. Mean results are reported.
<table><tr><td>Input Views</td><td>CD â CD-d â</td><td>CD-s â</td></tr><tr><td>16</td><td>5.32</td><td>16.71 10.25</td></tr><tr><td>8</td><td>8.61</td><td>23.41 41.54</td></tr><tr><td>4</td><td>9.60</td><td>25.87 31.02</td></tr><tr><td>3</td><td>15.40</td><td>41.23 60.37</td></tr><tr><td>2</td><td>72.08</td><td>118.63 168.37</td></tr></table>

Ablation of The Number of Input Views. To verify the performance of our method for different numbers of input views, we conduct ablation experiments on the PartNet-Mobility dataset. Note that all input views are perceived by the optimal view perception method. The experimental results in Table. 6 prove that our method can enhance the performance of mesh reconstruction by using more input views. Moreover, we observe that when the input views are less than 4, the reconstruction quality suffers a sharp decline. This is because the information provided by too few views is insufficient, and high-fidelity mesh reconstruction through the insufficient image supervision poses a significant challenge.

Ablation of Key Components. We conduct ablation of proposed core components on PartNet-Mobility dataset. In the âw/o planar Gaussiansâ setting, we employ the depth rendering pipeline from GaussianObject. In âw/o pseudo labelsâ setting, we omit the pseudo labels generated from reliable regions. The experimental results in Table. 7 demonstrate that each component makes a significant improvement to both whole surface and part-level reconstruction performance, especially for the planar Gaussians and the refinement. This confirms the effectiveness of proposed key components, and the superiority of planar Gaussians as well as the refinement via few-shot diffusion.

Table 7. Ablation of key components on PartNet-Mobility dataset. Mean results are reported.
<table><tr><td>Settings</td><td>CD â CD-d â CD-sâ</td></tr><tr><td>w/o depth smooth regularization</td><td>11.32 31.84 54.81</td></tr><tr><td>w/o depth regularization</td><td>15.84 34.20 59.32</td></tr><tr><td>w/o pseudo labels</td><td>10.27 30.58 51.08</td></tr><tr><td>w/o view regularization</td><td>10.18 30.08 48.63</td></tr><tr><td>w/o refinement</td><td>75.63 81.05 90.47</td></tr><tr><td>w/o planar Gaussians</td><td>30.98 61.52 76.25</td></tr><tr><td>w/ all</td><td>9.60 25.87 31.02</td></tr></table>

## 4.4. Training Time Analysis

On a single RTX 4090 GPU, our method takes about 45 minutes for the object-level reconstruction, which is comparable to GaussianObject. Concretely, the coarse-to-fine optimization takes about 40 minutes, making up most of the training time. The optimal viewpoint perception only requires about few minutes and representation initialization takes about 1 minute. The articulated modeling requires only 2 minutes approximately.

## 4.5. Generalization to the Real World

We present the quantitative results of real-world mesh reconstruction in Table. 4 and the corresponding qualitative results in Fig. 7. Our method significantly outperforms existing approaches in mean results across all metrics and demonstrates strong generalization on real-world data. As shown in Fig. 7, our method enables high-fidelity reconstruction for unseen real-world articulated objects with two or more parts.

## 4.6. Conclusion and Limitation

In this paper, we propose a category-agnostic articulated object reconstruction via planar Gaussians, which requires only single-stage sparse-view RGB images. We introduce an optimal sparse view perception approach through Gaussian information fields and perform a coarse-to-fine optimization as well as articulated modeling for planar Gaussians. Extensive experiments demonstrate our superiority compared to existing approaches.

The limitations of our method lie in the challenge of reconstructing articulated objects with transparent materials or extremely small size in real-world environments. Future work will focus on integrating physically based rendering to model transparent materials [5], and incorporating superresolution techniques [28] to improve reconstruction with low-resolution image inputs.

<!-- image-->  
Figure 7. The qualitative results of our real-world performance. Static parts are marked in blue, dynamic parts in yellow, and additional dynamic parts in red and green. Red arrows represent the joints.

## References

[1] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, and Guofeng Zhang. Pgsr: Planar-based gaussian splatting for efficient and high-fidelity surface reconstruction. arXiv preprint arXiv:2406.06521, 2024. 4, 5, 6, 7

[2] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping Liu, and Guosheng Lin. Gaussianeditor: Swift and controllable 3d editing with gaussian splatting, 2023. 5

[3] Junfu Guo, Yu Xin, Gaoyi Liu, Kai Xu, Ligang Liu, and Ruizhen Hu. Articulatedgs: Self-supervised digital twin modeling of articulated objects using 3d gaussian splatting. arXiv preprint arXiv:2503.08135, 2025. 2

[4] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. 4

[5] Letian Huang, Dongwei Ye, Jialin Dan, Chengzhi Tao, Huiwen Liu, Kun Zhou, Bo Ren, Yuanqi Li, Yanwen Guo, and Jie Guo. Transparentgs: Fast inverse rendering of transparent objects with gaussians. ACM Transactions on Graphics (TOG), 44(4):1â17, 2025. 9

[6] Zixuan Huang, Mark Boss, Aaryaman Vasishta, James M Rehg, and Varun Jampani. Spar3d: Stable point-aware reconstruction of 3d objects from single images. arXiv preprint arXiv:2501.04689, 2025. 2, 3

[7] Zhenyu Jiang, Cheng-Chun Hsu, and Yuke Zhu. Ditto: Building digital twins of articulated objects from interaction. In Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 1, 2

[8] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time

radiance field rendering. ACM Transactions on Graphics, 42 (4), 2023. 1, 2, 3, 4

[9] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, andÂ´ Ross Girshick. Segment anything. arXiv:2304.02643, 2023. 6

[10] Yang Li and Tatsuya Harada. Non-rigid point cloud registration with neural deformation pyramid. arXiv preprint arXiv:2205.12796, 2022. 4

[11] Jiayi Liu, Ali Mahdavi-Amiri, and Manolis Savva. Paris: Part-level reconstruction and motion analysis for articulated objects. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 352â363, 2023. 1, 2

[12] Jiayi Liu, Denys Iliash, Angel X Chang, Manolis Savva, and Ali Mahdavi-Amiri. SINGAPO: Single image controlled generation of articulated parts in object. arXiv preprint arXiv:2410.16499, 2024. 2

[13] Yu Liu, Baoxiong Jia, Ruijie Lu, Junfeng Ni, Song-Chun Zhu, and Siyuan Huang. Building interactable replicas of complex articulated objects via gaussian splatting. In The Thirteenth International Conference on Learning Representations, 2025. 1, 2, 6, 7

[14] Ruijie Lu, Yu Liu, Jiaxiang Tang, Junfeng Ni, Yuxiang Wang, Diwen Wan, Gang Zeng, Yixin Chen, and Siyuan Huang. Dreamart: Generating interactable articulated objects from a single image. arXiv preprint arXiv:2507.05763, 2025. 2

[15] Luca Medeiros. Language segment-anything: Sam with text prompt. https://github.com/luca-medeiros/ lang-segment-anything, 2024. Accessed: 2025-08- 31. 5

[16] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022. 5

[17] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 2

[18] Anish Mittal, Rajiv Soundararajan, and Alan C. Bovik. Making a âcompletely blindâ image quality analyzer. IEEE Signal Processing Letters, 20(3):209â212, 2013. 3

[19] Jiteng Mu, Weichao Qiu, Adam Kortylewski, Alan Yuille, Nuno Vasconcelos, and Xiaolong Wang. A-sdf: Learning disentangled signed distance functions for articulated shape representation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12981â12991, 2021. 1, 2

[20] Michael Niemeyer, Jonathan T. Barron, Ben Mildenhall, Mehdi S. M. Sajjadi, Andreas Geiger, and Noha Radwan. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2022. 2

[21] Avinash Paliwal, Wei Ye, Jinhui Xiong, Dmytro Kotovenko, Rakesh Ranjan, Vikas Chandra, and Nima Khademi Kalantari. Coherentgs: Sparse novel view synthesis with coherent

3d gaussians. In European Conference on Computer Vision, pages 19â37. Springer, 2024. 1, 2, 4, 5, 6, 7

[22] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image Â¨ synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684â10695, 2022. 4

[23] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 1

[24] Guangcong Wang, Zhaoxi Chen, Chen Change Loy, and Ziwei Liu. Sparsenerf: Distilling depth ranking for few-shot novel view synthesis. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 2

[25] Di Wu, Liu Liu, Zhou Linli, Anran Huang, Liangtu Song, Qiaojun Yu, Qi Wu, and Cewu Lu. Reartgs: Reconstructing and generating articulated objects via 3d gaussian splatting with geometric and motion constraints. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 1, 2, 6, 7

[26] Jiang Wu, Rui Li, Yu Zhu, Rong Guo, Jinqiu Sun, and Yanning Zhang. Sparse2dgs: Geometry-prioritized gaussian splatting for surface reconstruction from sparse views. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11307â11316, 2025. 2, 5, 6, 7

[27] Fanbo Xiang, Yuzhe Qin, Kaichun Mo, Yikuan Xia, Hao Zhu, Fangchen Liu, Minghua Liu, Hanxiao Jiang, Yifu Yuan, He Wang, et al. Sapien: A simulated part-based interactive environment. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11097â 11107, 2020. 6

[28] Shiyun Xie, Zhiru Wang, Xu Wang, Yinghao Zhu, Chengwei Pan, and Xiwang Dong. Supergs: Super-resolution 3d gaussian splatting enhanced by variational residual features and uncertainty-augmented learning. arXiv preprint arXiv:2410.02571, 2024. 9

[29] Yuhan Xie, Yixi Cai, Yinqiang Zhang, Lei Yang, and Jia Pan. Gauss-mi: Gaussian splatting shannon mutual information for active 3d reconstruction. arXiv preprint arXiv:2503.02881, 2025. 3

[30] Haolin Xiong, Sairisheek Muttukuru, Rishi Upadhyay, Pradyumna Chari, and Achuta Kadambi. Sparsegs: Realtime 360Â° sparse view synthesis using gaussian splatting, 2023. 2

[31] Chen Yang, Sikuang Li, Jiemin Fang, Ruofan Liang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian. Gaussianobject: High-quality 3d object reconstruction from four views with gaussian splatting. ACM Transactions on Graphics, 2024. 1, 2, 4, 5, 6, 7

[32] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In CVPR, 2024. 4

[33] Chi Zhang, Yujun Cai, Guosheng Lin, and Chunhua Shen. Deepemd: Differentiable earth moverâs distance for few-shot learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(5):5632â5648, 2022. 6

[34] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023. 4

[35] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 3

[36] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang. Fsgs: Real-time few-shot view synthesis using gaussian splatting, 2023. 1, 2