# WorldWarp: Propagating 3D Geometry with Asynchronous Video Diffusion

Hanyang Kong1 Xingyi Yang2\* Xiaoxu Zheng1 Xinchao Wang1\*

1National University of Singapore 2The Hong Kong Polytechnic University hanyang.k@u.nus.edu, xingyi.yang@polyu.edu.hk, xinchao@nus.edu.sg https://hyokong.github.io/worldwarp-page/

<!-- image-->  
Starting Image

1st Frame

3DGS Reconstruction

Figure 1. WorldWarp: Long-range novel view synthesis from a single image. Given only a single starting image (left) and a specified camera trajectory, our method generates a long and coherent video sequence. The core of our approach is to generate the video chunk-bychunk, where each new chunk is conditioned on forward-warped âhints" from the previous one. A novel diffusion model then generates the next chunk by correcting these hints and filling in occlusions using a spatio-temporal varying noise schedule. The high geometric consistency of our 200-frame generated sequence is demonstrated by its successful reconstruction into a high-fidelity 3D Gaussian Splatting (3DGS) [25] model (right). This highlights our modelâs robust understanding of 3D geometry and its capability to maintain long-term consistency.

## Abstract

Generating long-range, geometrically consistent video presents a fundamental dilemma: while consistency demands strict adherence to 3D geometry in pixel space, stateof-the-art generative models operate most effectively in a camera-conditioned latent space. This disconnect causes current methods to struggle with occluded areas and complex camera trajectories. To bridge this gap, we propose WorldWarp, a framework that couples a 3D structural anchor with a 2D generative refiner. To establish geometric grounding, WorldWarp maintains an online 3D geometric cache built via Gaussian Splatting (3DGS). By explicitly

warping historical content into novel views, this cache acts as a structural scaffold, ensuring each new frame respects prior geometry. However, static warping inevitably leaves holes and artifacts due to occlusions. We address this using a Spatio-Temporal Diffusion (ST-Diff) model designed for a "fill-and-revise" objective. Our key innovation is a spatio-temporal varying noise schedule: blank regions receive full noise to trigger generation, while warped regions receive partial noise to enable refinement. By dynamically updating the 3D cache at every step, WorldWarp maintains consistency across video chunks. Consequently, it achieves state-of-the-art fidelity by ensuring that 3D logic guides structure while diffusion logic perfects texture. Project page: https://hyokong.github.io/worldwarp-page/.

## 1. Introduction

Novel View Synthesis (NVS) has emerged as a cornerstone problem in computer vision and graphics, with transformative applications in virtual reality, immersive telepresence, and generative content creation. While traditional NVS methods excel at view interpolation, which generates new views within the span of existing camera poses [2, 25, 40], the frontier of the field lies in view extrapolation [16, 32, 33, 37, 55, 67]. This far more challenging task involves generating long, continuous camera trajectories that extend significantly beyond the original scene, effectively synthesizing substantial new content and structure [37, 55]. The ultimate goal is to enable interactive exploration of dynamic, 3D-consistent worlds from only a limited set of starting images.

The central challenge in generating long-range, cameraconditioned video lies in finding an effective 3D conditioning. Existing works have largely followed two main strategies. The first is camera pose encoding, which embeds abstract camera parameters as a latent condition [16, 29, 39, 54, 55, 67]. This approach, however, relies heavily on the diversity of the training dataset and often fails to generalize to Out-Of-Distribution (OOD) camera poses, while also providing minimal information about the underlying 3D scene content [11, 20, 32, 41, 44, 77]. The second strategy, which uses an explicit 3D spatial prior, was introduced to solve this OOD issue [11, 20, 32, 77]. While these priors provide robust geometric grounding, they are imperfect, suffering from occlusions (blank regions) and distortions from 3D estimation errors [55, 77]. This strategy typically employs standard inpainting or video generation techniques [11, 20, 32], which are ill-suited to simultaneously handle the severe disocclusions and the geometric distortions present in the warped priors, leading to artifacts and inconsistent results.

To address this critical gap, we propose WorldWarp, a novel framework that generates long-range, geometricallyconsistent novel view sequences. Our core insight is to break the strict causal chain of AR models and the static nature of explicit 3D priors. Instead, WorldWarp operates via an autoregressive inference pipeline that generates video chunkby-chunk (see Fig. 3). The key to our system is a Spatio-Temporal Diffusion (ST-Diff) model [49, 60], which is trained with a powerful bidirectional, non-causal attention mechanism. This non-causal design is explicitly enabled by our core technical idea: using forward-warped images from future camera positions as a dense, explicit 2D spatial prior [9]. At each step, we build an "online 3D geometric cache" using 3DGS [25], which is optimized only on the most recent, high-fidelity generated history. This cache then renders high-quality warped priors for the next chunk, providing ST-Diff with a rich, geometrically-grounded signal that guides the generation of new content and fills occlusions.

The primary advantage of WorldWarp is its ability to avoid the irreversible error propagation that plagues prior work [55, 77]. By dynamically re-estimating a short-term 3DGS cache at each step, our method continuously grounds itself in the most recent, accurate geometry, ensuring highfidelity consistency over extremely long camera paths. We demonstrate the effectiveness of our approach through extensive experiments on challenging real-world and synthetic datasets for long-sequence view extrapolation, achieving state-of-the-art performance in both geometric consistency and visual fidelity.

In summary, our main contributions are:

â¢ WorldWarp, a novel framework for long-range novel view extrapolation that generates video chunk-by-chunk using an autoregressive inference pipeline.

â¢ Spatio-Temporal Diffusion (ST-Diff), a non-causal diffusion model that leverages bidirectional attention conditioned on forward-warped images as a dense geometric prior.

â¢ An online 3D geometric cache mechanism, which uses test-time optimized 3DGS [25] to provide high-fidelity warped priors while preventing the irreversible error propagation of static 3D representations.

â¢ State-of-the-art performance on challenging view extrapolation benchmarks, demonstrating significantly improved geometric consistency and image quality over existing methods.

## 2. Related Works

Novel view synthesis. Novel view synthesis (NVS) is a challenging problem that can be categorized into two aspects: view interpolation [2, 3, 25, 31, 40, 42, 46, 51, 52, 62, 71, 78, 79, 85] and extrapolation [16, 30, 32, 33, 37, 48, 55, 67, 76, 83]. View interpolation task aims to generate novel views within the distributions of the training views [2, 3, 25, 40, 78] even if the training views are sparse [31, 42, 62, 85] or the training views are captured in the wild with occlusions [46, 51, 52]. View extrapolation tasks [16, 30, 32, 33, 37, 48, 55, 67, 76, 83] focus on generating novel views which are extended significantly beyond the original scenes, introducing substantial new contents, by leveraging powerful pre-trained video diffusion models [21, 35, 49, 60, 73].

Auto-regressive video diffusion models. The field of video generation has seen a prominent trend towards either diffusion-based or autoregressive (AR) methodologies. Parallel (non-autoregressive) video diffusion systems often employ bidirectional attention to process and denoise all frames concurrently [4â6, 10, 14, 15, 18, 19, 28, 43, 59, 61, 74]. Conversely, AR-based techniques produce content in a sequential manner. This category encompasses several architectures, such as models based on pure next-token prediction [7, 23, 27, 45, 65, 68, 72], more recent hybrid systems integrating AR and diffusion principles [8, 12, 13, 22, 24, 34, 38, 69, 75, 82], and rolling diffusion variants that employ progressive noise schedules [26, 50, 53, 58, 70, 80]. However, these AR strategies are ill-suited for this workâs specific task. Learning an effective camera embedding for them is non-trivial, and their causal structure is incompatible with using warped images from future camera positions as conditional hints. Consequently, this work employs a non-autoregressive framework [57] to leverage this future information.

Camera pose encoding and 3D explicit spatial priors. Spatially consistent view generation relies on conditioning. One method, camera pose encoding, models camera geometry using absolute extrinsics [16, 39, 54, 55, 67] or relative representations like CaPE [29]. While useful for viewpoint control, these encodings lack 3D scene content. An alternative, explicit 3D spatial priors, builds 3D models (e.g., meshes, point clouds, 3DGS [25]) [11, 20, 32, 41, 44, 77] for re-projection and inpainting. This provides geometric grounding but suffers from error propagation from the initial 3D estimation [55, 77] and high computational cost. Instead, we utilize forward-warped images from future camera positions as a distinct explicit prior. These warped images serve as a dense, geometrically-grounded 2D hint, bypassing the error-prone and costly 3D reconstruction pipeline while offering a richer conditional signal than mere pose encoding.

## 3. Preliminaries

## 3.1. Camera-Conditioned Video Generation

One major challenge in adding precise camera control to video diffusion models is finding a good way to represent 3D camera movement. Simply using raw camera intrinsics K and extrinsics E is often suboptimal, as their numerical values (e.g., translation t) are unconstrained and difficult for a network to correlate with visual content.

A more effective paradigm is to translate these abstract parameters into a dense, pixel-wise representation that provides a clearer geometric interpretation. For example, PlÃ¼cker embeddings [56] define a 6D ray vector for each pixel. This transforms the abstract matrices into a dense tensor $\mathbf { P } \in \mathbb { R } ^ { n \times 6 \times h \times w }$ , which is much more informative for the diffusion model. This principle of using dense, geometrically-grounded priors is a key consideration for enabling fine-grained camera control.

## 3.2. Diffusion Forcing and Non-Causal Priors

The Diffusion Forcing Transformer (DFoT) [57] paradigm reframes the noising operation as progressive masking, where each frame $\mathbf { x } _ { t }$ in a video is assigned an independent noise level $k _ { t } \in [ 0 , 1 ]$ . This contrasts with conventional models that use a single noise level k for all frames. The model $\hat { \epsilon } _ { \theta }$ is then trained on a per-frame noise prediction loss:

$$
\mathcal { L } = \underset { k _ { T } , \mathcal { X } , \mathcal { \mathcal { E } } } { \mathbb { E } } \Big [ \sum _ { t = 1 } ^ { T } \big \lVert \hat { \epsilon } _ { \boldsymbol { \theta } } ( \mathcal { X } ^ { k } , k _ { T } ) _ { t } - \epsilon _ { t } \big \rVert _ { 2 } ^ { 2 } \Big ]\tag{1}
$$

The critical advantage of this per-frame noise approach is that it enables a model to be trained with non-causal attention, learning to denoise a frame by conditioning on an arbitrary, partially-masked set of other frames.

This non-causal paradigm is particularly well-suited for our task. In typical video generation, a causal architecture is necessary as the future is unknown. However, in cameraconditioned novel view synthesis, we can generate a strong, geometry-consistent prior for all future frames simultaneously via forward-warping. These warped images provide a powerful non-causal conditioning signal. This insight is the foundation of our ST-Diff model, allowing us to discard restrictive causal constraints and employ a bidirectional, spatio-temporal diffusion strategy.

## 4. Method

## 4.1. Spatio-Temporal Diffusion with Warped Priors

We address the task of novel view synthesis, where the goal is to generate a target view $\mathbf { x } _ { t }$ given a source view $\mathbf { x } _ { s }$ and corresponding camera poses $\left\{ \mathbf { p } _ { s } , \mathbf { p } _ { t } \right\}$ . To this end, we introduce Spatio-Temporal Diffusion with Warped Priors (ST-Diff), a bidirectional diffusion model designed for this task. Unlike causal, autoregressive video generation, where future frames are unknown, the camera-conditioned setting allows us to form a strong geometric prior for the target frame by projecting the source view. This key insight allows us to discard causal constraints and employ a more powerful bidirectional attention mechanism across all frames.

Our method first prepares geometric priors in the pixel space and then performs all diffusion, compositing, and noising operations in the latent space using a pre-trained VAE encoder $\mathcal { E } ( \cdot )$ and decoder D(Â·) [49, 60]. We use x to denote data in pixel-space and z for latent-space data.

One-to-all pixel-space warping. Given a training video sequence ${ \mathcal X } = \{ { \bf x } _ { i } \} _ { i = 1 } ^ { T }$ , we first sample a single source frame $\mathbf { x } _ { s }$ from the sequence. We then create a full sequence of warped priors by warping this single source frame xs to every other frameâs viewpoint, including its own. To do this, we use pre-estimated depth maps $\mathbf { D } _ { i }$ and camera parameters (extrinsics $\mathbf { E } _ { i }$ and intrinsics $\mathbf { K } _ { i } )$ for all frames, obtained from a 3D geometry foundation model [9]. First, the source image $\mathbf { x } _ { s }$ and its depth $\mathbf { D } _ { s }$ are unprojected into a 3D RGB point cloud $\mathcal { P } _ { s }$ :

$$
\mathbf { p } _ { \mathrm { c a m } } ^ { ( u , v ) } = \mathbf { D } _ { s } ( u , v ) \cdot \mathbf { K } _ { s } ^ { - 1 } [ u , v , 1 ] ^ { T }\tag{2}
$$

$$
\mathcal { P } _ { s } = \{ ( \mathbf { E } _ { s } \mathbf { p } _ { \mathrm { c a m } } ^ { ( u , v ) } , \mathbf { x } _ { s } ( u , v ) ) \} _ { u , v }\tag{3}
$$

This single point cloud $\mathcal { P } _ { s }$ is then rendered into all $T$ target viewpoints using a differentiable point-based renderer. This "one-to-all" warping process yields two new sequences: a warped prior sequence, $\mathscr X _ { s \to \mathscr V } = \{ \mathbf x _ { s \to t } \} _ { t = 1 } ^ { T }$ , and a corresponding validity mask sequence, $\mathcal { M } = \{ \mathbf { M } _ { t } \} _ { t = 1 } ^ { T }$ . Each mask Mt indicates which pixels in $\mathbf { x } _ { s  t }$ were successfully rendered from $\mathcal { P } _ { s }$

<!-- image-->  
Figure 2. Training pipeline of our ST-Diff model. 1) Spatially temporally-varying noisy latent: The process begins by rendering a warped image and a validity mask from an RGB point cloud (images are shown for illustration, as operations are in latent space). The warped image is encoded to get $\mathbf { z } _ { s  t }$ , and the ground-truth image is encoded to get $\mathbf { z } _ { t } , \mathbf { A }$ "clean composite" latent ${ \bf z } _ { c , t }$ is created by combining the valid warped regions from $\mathbf { z } _ { s  t }$ with the blank regions from $\mathbf { z } _ { t } .$ , using the downsampled mask $\mathbf { M } _ { \mathrm { l a t e n t } }$ . 2) Training ST-Diffusion: This composite latent sequence is noised according to our spatio-temporal schedule, resulting in a noisy latent sequence (visualized as a stack) where the noise level for each latent varies across different frames and spatial regions. The resulting noisy latents are fed into our model $G _ { \theta } .$ which is trained to predict the target velocity (defined as $\epsilon _ { t } - \mathbf { z } _ { t } )$ , forcing it to learn the flow from the noisy composite latent back towards the original ground-truth latent sequence $\mathcal { Z } .$

Latent-space composite sequence. The training pipeline of our WorldWarp is illustrated in Fig. 2. With the pixelspace assets prepared, we move entirely to the latent space. We separately encode the new warped sequence $\mathcal { X } _ { s  \nu }$ and the original ground-truth sequence X . We encode both:

$$
\mathcal { Z } _ { s  \mathcal { V } } = \{ \mathcal { E } ( \mathbf { x } _ { s  t } ) \} _ { t = 1 } ^ { T } \quad \mathrm { a n d } \quad \mathcal { Z } = \{ \mathcal { E } ( \mathbf { x } _ { t } ) \} _ { t = 1 } ^ { T } .\tag{4}
$$

We also downsample the mask sequence M to match the latent dimensions, yielding $\mathcal { M } _ { \mathrm { l a t e n t } } = \{ \mathbf { M } _ { \mathrm { l a t e n t } , t } \} _ { t = 1 } ^ { T } . \mathrm { ~ A ~ }$ clean composite latent sequence $\mathcal { Z } _ { c }$ is then created in the latent space. For each frame $t ,$ the composite $\mathbf { z } _ { c , t }$ takes its features from the warped latent $\mathbf { z } _ { s  t }$ in valid ("warped") regions and fills the remaining ("filled") regions with features from the ground-truth latent $\mathbf { z } _ { t }$ (which is the t-th element of Z ):

$$
\mathbf { z } _ { c , t } = \mathbf { M } _ { \mathrm { l a t e n t } , t } \odot \mathbf { z } _ { s  t } + ( 1 - \mathbf { M } _ { \mathrm { l a t e n t } , t } ) \odot \mathbf { z } _ { t } \quad \mathrm { f o r } t = 1 . . . T\tag{5}
$$

This entire sequence $\mathcal { Z } _ { c } ~ = ~ \{ \mathbf { z } _ { c , t } \} _ { t = 1 } ^ { T }$ serves as the $x _ { 0 ^ { - } }$ equivalent (clean signal) for the diffusion model.

Spatially and temporally-varying noise. Our noising strategy extends the per-frame independent noise concept with a new, region-specific dimension, as shown in Fig. 2. The noise applied is varied at two levels simultaneously. First, at a temporal level, each frame t in the sequence $\mathcal { Z } _ { c }$ gets a different, independently sampled noise schedule. Second, at a spatial level, we apply different noise levels \*within\* each frame, distinguishing between the "warped" and "filled" regions. For each frame t, we therefore sample a pair of noise levels, $( \sigma _ { \mathrm { w a r p e d } , t } , \sigma _ { \mathrm { f i l l e d } , t } )$ . A spatially-varying noise map $\Sigma _ { t }$ is constructed using the latent-space mask:

$$
\pmb { \Sigma } _ { t } = \mathbf { M } _ { \mathrm { l a t e n t } , t } \odot \sigma _ { \mathrm { w a r p e d } , t } + ( 1 - \mathbf { M } _ { \mathrm { l a t e n t } , t } ) \odot \sigma _ { \mathrm { f i l l e d } , t }\tag{6}
$$

We then generate the final noisy input sequence $\mathcal { Z } _ { \mathrm { n o i s y } } =$ $\{ \mathbf { z } _ { \mathrm { n o i s y } , t } \} _ { t = 1 } ^ { T }$ by sampling a noise sequence $\mathcal { E } = \{ \epsilon _ { t } \} _ { t = 1 } ^ { T } \sim$ $\mathcal { N } ( 0 , \bf { I } )$

$$
\mathbf { z } _ { \mathrm { n o i s y } , t } = \left( 1 - \Sigma _ { t } \right) \odot \mathbf { z } _ { c , t } + \Sigma _ { t } \odot \boldsymbol { \epsilon } _ { t }\tag{7}
$$

A key architectural modification is required to process this spatiallyand temporally varying noise. Standard diffusion models [60] typically accept a single timestep embedding (e.g., shape $B \times 1 )$ for an entire image or video chunk. Our ST-Diff model, however, is adapted to process a unique noise level for every token. We broadcast the noise map sequence $\begin{array} { r } { \Sigma _ { \mathcal { V } } = \{ \Sigma _ { t } \} _ { t = 1 } ^ { T } } \end{array}$ to the full latent sequence dimensions $( B \times$ $T \times H ^ { \prime } \times W ^ { \prime } )$ and pass it through the time embedding network, thus generating a unique time-axis and spatial-axis embedding for each corresponding token.

Training objective. We train our ST-Diff model $G _ { \theta }$ which takes the entire noisy sequence $\mathcal { Z } _ { \mathrm { n o i s y } } .$ , the sequence of noise maps $\Sigma _ { \mathcal { V } }$ , and other conditioning c (e.g., text, camera poses) as input. Critically, the model is trained to denoise the composite sequence $\mathcal { Z } _ { \mathrm { n o i s y } }$ while regressing towards a target defined by the original ground-truth latent sequence $\mathcal { Z } .$ The target velocity sequence is $\mathcal { V } _ { \mathrm { t a r g e t } } = \{ \epsilon _ { t } - \mathbf { z } _ { t } \} _ { t = 1 } ^ { T }$ . Our training objective is the $L _ { 2 }$ loss, summed over the entire sequence:

<!-- image-->  
Figure 3. The autoregressive inference pipeline of WorldWarp. At each iteration k, the available history (either the initial images or the previously generated k â 1 chunk) is processed. First, TTT3R estimates camera poses and an initial 3D point cloud. This geometry is used to optimize a 3D Gaussian Splatting (3DGS) representation, which serves as a high-fidelity 3D cache. Concurrently, a VLM generates a descriptive text prompt, and novel camera poses are extrapolated for the next chunk. The optimized 3DGS renders forward-warped images at these new poses. These warped priors, along with the VLM prompt, are fed into our non-causal ST-Diff model $\left( G _ { \theta } \right)$ to denoise and generate the k-th chunk of novel views. The process then repeats, using the newly generated chunk as the history for the next iteration.

$$
\mathcal { L } = \mathbb { E } _ { \mathcal { Z } , \mathcal { Z } _ { c } , \mathcal { E } , \Sigma _ { \mathcal { V } } , \mathbf { c } } \left[ \sum _ { t = 1 } ^ { T } \left. \mathbf { v } _ { \boldsymbol { \theta } , t } - \left( \epsilon _ { t } - \mathbf { z } _ { t } \right) \right. _ { 2 } ^ { 2 } \right]\tag{8}
$$

where $\mathcal { V } _ { \theta } ~ = ~ \{ \mathbf { v } _ { \theta , t } \} _ { t = 1 } ^ { T } ~ = ~ G _ { \theta } ( \mathcal { Z } _ { \mathrm { n o i s y } } , \Sigma _ { \mathcal { V } } , \mathbf { c } )$ . This loss forces the model to learn the complex relationship between the warped, GT-filled, and final target latents across the entire video.

## 4.2. Autoregressive Inference Pipeline

The inference process is illustrated in Fig. 3. Our inference process generates novel view sequences autoregressively, producing a video chunk-by-chunk in a for-loop manner. Unlike training, which uses a fixed-radius point cloud representation, our inference pipeline leverages a dynamic, testtime optimized 3D representation as an explicit geometric cache. This process, illustrated in Fig. 3, integrates 3D Gaussian Splatting [25] (3DGS) for high-fidelity warping and a Vision-Language Model (VLM) [1] for semantic guidance.

Online 3D Geometric Cache. At the beginning of each iteration k of the generation loop, we take the available history (either the initial source views for k = 1 or the video chunk generated in the previous iteration k â 1). We first process these frames using a 3D geometry model (TTT3R) [9] to estimate their camera poses and an initial 3D point cloud. This point cloud is then used to initialize a 3D Gaussian Splatting (3DGS) representation, which we optimize for a few hundred steps (e.g., 200 steps) using the history frames and their estimated poses. This resulting online-optimized 3DGS model serves as an explicit, high-fidelity 3D representation cache. Compared to the fixed-radius point clouds used during training, this 3DGS provides significantly higher-quality features for the non-blank (warped) regions, which is critical for maintaining geometric consistency.

Chunk-based Generation with ST-Diff. With the geometric and semantic conditioning prepared, we first render the sequence of prior images, $\mathcal { X } _ { s  \nu }$ , from the 3DGS cache. These are encoded into latents $\mathcal { Z } _ { s  \mathcal { V } } = \{ \mathbf { z } _ { s  t } \} _ { t = 1 } ^ { T }$ and we also obtain the corresponding latent-space masks $\mathcal { M } _ { \mathrm { l a t e n t } } = \{ \mathbf { M } _ { \mathrm { l a t e n t } , t } \} _ { t = 1 } ^ { T }$ . Our goal is twofold: to fill in the blank (occluded) regions and to revise the non-blank (warped) regions, which may suffer from blur or distortion.

We achieve this by initializing the reverse diffusion process from a spatially-varying noise level, analogous to imageto-video translation. Let the full reverse schedule consist of N timesteps, from $T _ { N } = 1 0 0 0$ down to $T _ { 1 } = 1$ . We define a strength parameter $\tau \in [ 0 , 1 ]$ , which maps to an intermediate timestep $T _ { \mathrm { s t a r t } }$ and its corresponding noise level $\sigma _ { \mathrm { s t a r t } }$ . We set the noise level for the blank (filled) regions to $\sigma _ { \mathrm { f i l l e d } } = \sigma _ { T _ { N } }$ , which corresponds to pure noise.

For each frame t, we construct a spatially-varying noise map $\Sigma _ { \mathrm { s t a r t } , t }$ using the latent-space mask:

$$
\pmb { \Sigma } _ { \mathrm { s t a r t } , t } = \mathbf { M } _ { \mathrm { l a t e n t } , t } \odot \sigma _ { \mathrm { s t a r t } } + ( 1 - \mathbf { M } _ { \mathrm { l a t e n t } , t } ) \odot \sigma _ { \mathrm { f i l l e d } }\tag{9}
$$

We then generate the initial noisy latent sequence $\mathcal { Z } _ { \mathrm { s t a r t } } =$ $\{ \mathbf { z } _ { \mathrm { s t a r t } , t } \} _ { t = 1 } ^ { T }$ for the reverse process. This is done by applying the noise map $\Sigma _ { \mathrm { s t a r t } , t }$ to the warped latent $\mathbf { z } _ { s  t } .$ , using a sampled Gaussian noise $\epsilon _ { t } \dot { \cdot }$

$$
\mathbf { z } _ { \mathrm { s t a r t } , t } = ( 1 - \Sigma _ { \mathrm { s t a r t } , t } ) \odot \mathbf { z } _ { s  t } + \Sigma _ { \mathrm { s t a r t } , t } \odot \boldsymbol { \epsilon } _ { t }\tag{10}
$$

This formulation effectively initializes the blank regions with pure noise (as $\sigma _ { \mathrm { f i l l e d } } \approx 1 . 0 )$ while applying a partial, strength-controlled noising to the warped regions.

<!-- image-->  
Figure 4. Qualitative comparisons on the RealEstate10K [84] and DL3DV [36] datasets. We visualize videos generated by our method against those by GenWarp [55], CameraCtrl [16], and VMem [32]. Our WorldWarp generalizes to diverse camera motion, showcasing the spatial and temporal consistency.

Our ST-Diff model $( G _ { \theta } )$ then takes this spatially-mixed latent sequence ${ \mathcal { Z } } _ { \mathrm { s t a r t } } .$ , the VLM text prompt, and the corresponding spatially-varying time embeddings as input. It denoises the sequence beginning from its spatially-varying timesteps (e.g., $T _ { \mathrm { s t a r t } }$ for warped regions and $T _ { N }$ for blank regions) down to $T _ { 1 }$ to generate the k-th chunk of novel views. This newly generated chunk is then used as the history for the next iteration (k + 1), and the entire process repeats.

## 5. Experiments

## 5.1. Implementation Details.

We fine-tune WorldWarp based on Wan2.1-T2V-1.3B [60] model, with resolution 720x480 and batch size 8, on 8 H200 GPUs for 10k iterations. We apply TTT3R [9] as the 3D reconstruction foundation model for estimating camera parameters and depth maps. Please refer to the supplementary material for more details.

Datasets and evaluation metrics. We conduct experiments on two public scene-level datasets: RealEstate10K (Re10K) [84] and DL3DV [36] datasets. Our evaluation of novel view synthesis quality comprises three main components: 1) Perceptual quality: We measure the distributional similarity between generated views and the test set using the FrÃ©chet Image Distance (FID) [17]. 2) Detail preservation: Following [47], we assess the modelâs ability to preserve image details across views by computing PSNR, SSIM [66], and LPIPS [81]. 3) Geometric alignment: We evaluate camera pose accuracy against the ground truth $( \mathbf { R } _ { \mathrm { g t } } ,$ $\mathbf { t } _ { \mathrm { g t } } )$ , following [67]. We use DUST3R [64] to extract poses $( \mathbf { R } _ { \mathrm { g e n } } , \mathbf { t } _ { \mathrm { g e n } } )$ from generated views. We then compute the rotation distance $( R _ { \mathrm { d i s t } } )$ and translation distance (tdist):

$$
\begin{array} { r l } & { R _ { \mathrm { d i s t } } = \operatorname { a r c c o s } \left\{ 0 . 5 ( \operatorname { t r } ( \mathbf { R } _ { \mathrm { g e n } } \mathbf { R } _ { \mathrm { g t } } ^ { T } ) - 1 ) \right\} } \\ & { \ t _ { \mathrm { d i s t } } = \| \mathbf { t } _ { \mathrm { g t } } - \mathbf { t } _ { \mathrm { g e n } } \| _ { 2 } , } \end{array}
$$

where tr stands for the trace of a matrix. Per [16], estimated poses are expressed relative to the first frame, and translation is normalized by the furthest frame.

## 5.2. Comparisons on the RealEstate10K Dataset

We present a comprehensive quantitative evaluation on the RealEstate10K dataset in Table 1, assessing generation quality (PSNR, LPIPS) and camera pose accuracy $( R _ { \mathrm { d i s t } } , T _ { \mathrm { d i s t } } )$

Table 1. Quantitative comparison for single-view NVS on the RealEstate10K [84] dataset. We report performance for both short-term $( 5 0 ^ { t h }$ frame) and long-term $( 2 0 0 ^ { t h }$ frame) synthesis. For each metric, the best , second best , and third best results are highlighted. Our method significantly outperforms all baselines across most metrics, demonstrating superior quality and temporal consistency.
<table><tr><td rowspan="2"></td><td colspan="6">Short-term (50thframe)</td><td colspan="6">Long-term (200th frame)</td></tr><tr><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td> $\mathbf { F I D \downarrow }$ </td><td> $R _ { \mathrm { d i s t } } \downarrow$ </td><td> $T _ { \mathrm { d i s t } } \downarrow$ </td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>FID â</td><td> $R _ { \mathrm { d i s t } } \downarrow$ </td><td> $T _ { \mathrm { d i s t } } \downarrow$ </td></tr><tr><td>InfiniteNature [37]</td><td>14.12</td><td>0.192</td><td>0.428</td><td>27.35</td><td>0.988</td><td>0.521</td><td>10.07</td><td>0.109</td><td>0.658</td><td>46.12</td><td>1.234</td><td>0.902</td></tr><tr><td>InfiniteNature-Zero [33]</td><td>14.31</td><td>0.201</td><td>0.409</td><td>26.98</td><td>0.931</td><td>0.502</td><td>10.22</td><td>0.117</td><td>0.647</td><td>45.71</td><td>1.201</td><td>0.885</td></tr><tr><td>GeoGPT [48]</td><td>13.54</td><td>0.186</td><td>0.437</td><td>26.52</td><td>0.732</td><td>0.413</td><td>9.67</td><td>0.089</td><td>0.664</td><td>41.52</td><td>1.112</td><td>0.876</td></tr><tr><td>Lookout [47]</td><td>14.63</td><td>0.216</td><td>0.372</td><td>28.67</td><td>1.221</td><td>0.864</td><td>10.73</td><td>0.163</td><td>0.647</td><td>57.82</td><td>1.331</td><td>0.912</td></tr><tr><td>PhotoNVS [76]</td><td>15.76</td><td>0.247</td><td>0.324</td><td>21.62</td><td>0.735</td><td>0.464</td><td>11.76</td><td>0.176</td><td>0.573</td><td>41.67</td><td>1.273</td><td>0.628</td></tr><tr><td>GenWarp [55]</td><td>13.21</td><td>0.252</td><td>0.428</td><td>29.51</td><td>0.553</td><td>0.059</td><td>9.72</td><td>0.192</td><td>0.601</td><td>36.12</td><td>1.136</td><td>0.446</td></tr><tr><td>MotionMotrl [67]</td><td>14.14</td><td>0.258</td><td>0.327</td><td>19.12</td><td>0.336</td><td>0.353</td><td>9.26</td><td>0.187</td><td>0.593</td><td>35.21</td><td>1.134</td><td>0.697</td></tr><tr><td>CameraCtrl [16]</td><td>14.97</td><td>0.271</td><td>0.311</td><td>20.07</td><td>0.308</td><td>0.267</td><td>11.16</td><td>0.183</td><td>0.584</td><td>35.07</td><td>1.206</td><td>0.704</td></tr><tr><td>ViewCrafter [77]</td><td>17.23</td><td>0.279</td><td>0.367</td><td>22.21</td><td>1.242</td><td>0.201</td><td>9.96</td><td>0.157</td><td>0.578</td><td>33.82</td><td>1.571</td><td>0.814</td></tr><tr><td>SEVA [83]</td><td>18.67</td><td>0.394</td><td>0.281</td><td>17.14</td><td>0.259</td><td>0.116</td><td>13.24</td><td>0.227</td><td>0.443</td><td>28.47</td><td>1.112</td><td>0.731</td></tr><tr><td>VMem [32]</td><td>18.19</td><td>0.403</td><td>0.273</td><td>16.97</td><td>0.221</td><td>0.043</td><td>14.91</td><td>0.223</td><td>0.471</td><td>25.17</td><td>1.132</td><td>0.494</td></tr><tr><td>DFoT [57]</td><td>18.53</td><td>0.439</td><td>0.265</td><td>17.27</td><td>0.326</td><td>0.318</td><td>15.21</td><td>0.245</td><td>0.418</td><td>24.85</td><td>1.643</td><td>0.835</td></tr><tr><td>Ours</td><td>20.32</td><td>0.527</td><td>0.216</td><td>15.56</td><td>0.188</td><td>0.039</td><td>17.13</td><td>0.281</td><td>0.352</td><td>21.75</td><td>0.697</td><td>0.203</td></tr></table>

for short-term $( 5 0 ^ { t h }$ frame) and long-term $( 2 0 0 ^ { t h }$ frame) synthesis. Our method achieves state-of-the-art results, outperforming all baselines across all 12 metrics. This advantage is most pronounced in the challenging long-term setting: while most methods suffer significant quality degradation, our model maintains the highest PSNR (17.13) and LPIPS (0.352), surpassing strong competitors like SEVA, VMem, and DFoT. This high fidelity is crucial, as pose estimation (using Master3R) fails on the low-quality or blurry outputs from baselines. Consequently, our model achieves the lowest long-term pose error $( R _ { \mathrm { d i s t } } \ 0 . 6 9 7 , T _ { \mathrm { d i s t } } \ 0 . 2 0 3 )$ . This highlights a clear distinction: camera-embedding methods (MotionCtrl, CameraCtrl) suffer severe pose drift, and while 3D-aware methods (GenWarp, VMem) are more stable, our spatial-temporal noise diffusion strategy significantly surpasses both, proving its superior ability to mitigate cumulative camera drift. Qualitative results are in Fig. 4 and the supplementary.

## 5.3. Comparisons on the DL3DV Dataset

We further validate our model on the more challenging DL3DV dataset in Tab. 2. Despite the complex trajectories degrading performance for all methods, our model maintains a commanding lead in all 12 metrics, demonstrating superior robustness. In the demanding long-term $( 2 0 0 ^ { t h }$ frame) setting, our modelâs PSNR (14.53) decisively outperforms the next-best competitors, DFoT (13.51) and VMem (12.28). This fidelity is again proven critical for pose accuracy. On this complex dataset, our model remains the most stable, achieving the lowest $R _ { \mathrm { d i s t } } ~ ( 1 . 0 0 7 )$ and $T _ { \mathrm { d i s t } } ~ ( 0 . 4 1 2 )$ . The weaknesses of competing approaches are magnified here, as 3D-aware methods like GenWarp $( 1 . 3 5 1 R _ { \mathrm { d i s t } } )$ and VMem $( 1 . 4 1 9 R _ { \mathrm { d i s t } } )$ lose stability. This proves our spatial-temporal noise diffusion strategy is more effective at preserving 3D consistency and mitigating severe camera drift on complex, long-range trajectories. Visualizations are in Fig. 4 and the supplementary material.

<!-- image-->  
Figure 5. Illustration of the ST-Diffâs generating process. We illustrate the GT images, the warped images which serve as the condition for ST-Diff, the corresponding validity mask, and our final generated frames. The comparisons show that our ST-Diff successfully fills in the blank areas (initialized from a full noise level) while simultaneously revising distortions and enhancing details in the non-blank regions (initialized from a partial noise level) during the diffusion process.

## 5.4. Ablation Study

We conduct ablation studies on the RealEstate10K dataset in Table 3 to validate our two core design choices: the 3DGSbased cache and the spatial-temporal noise diffusion model. Caching Mechanism. We first analyze the effect of our caching module. The "No Cache" baseline, which relies only on the initial image, fails completely in long-term generation, with PSNR dropping to 9.22. This confirms the necessity of a 3D cache for long-range synthesis. We then compare our full model, "Caching by online optimized 3DGS," against "Caching by RGB point cloud." Although our model is trained on warped point clouds (with unoptimized, uniform radii), using a simple point cloud cache at inference ("Caching by RGB point cloud") yields significantly lower performance (11.12 PSNR) than our full model (17.13 PSNR). This demonstrates that using an online optimized 3DGS as the cache provides a much more robust and high-fidelity 3D representation. Notably, this 3DGS optimization is highly efficient, requiring only 500 steps per chunk. This result confirms that despite the modality gap between training (point clouds) and inference (3DGS), the superior representation quality of 3DGS leads to a substantial improvement in both generation quality and pose accuracy.

Table 2. Single-view NVS on DL3DV dataset [36] Short-term evaluation is on the $5 0 ^ { t h }$ frame, and long-term is on frames $2 0 0 ^ { t h }$ . This dataset is significantly more challenging due to complex camera trajectories and diverse environments. All methods show a noticeable performance drop compared to RealEstate10K [84]. For each metric, the best , second best , and third best results are highlighted.
<table><tr><td rowspan="2">Method</td><td colspan="6">Short-term  $( 5 0 ^ { t h } \mathrm { f r a m e } )$ </td><td colspan="6">Long-term (200thframe)</td></tr><tr><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>FID â</td><td> $R _ { \mathrm { d i s t } } \downarrow$ </td><td> $T _ { \mathrm { d i s t } } \downarrow$ </td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>FID â</td><td>Rdist â</td><td> $T _ { \mathrm { d i s t } } \downarrow$ </td></tr><tr><td>InfiniteNature [37]</td><td>10.05</td><td>0.112</td><td>0.662</td><td>51.45</td><td>1.478</td><td>1.131</td><td>8.98</td><td>0.100</td><td>0.695</td><td>54.32</td><td>1.561</td><td>1.522</td></tr><tr><td>InfiniteNature-Zero [33]</td><td>10.21</td><td>0.121</td><td>0.648</td><td>51.05</td><td>1.432</td><td>1.109</td><td>9.12</td><td>0.107</td><td>0.685</td><td>53.95</td><td>1.528</td><td>1.501</td></tr><tr><td>GeoGPT [48]</td><td>9.83</td><td>0.096</td><td>0.688</td><td>50.12</td><td>1.553</td><td>1.407</td><td>8.52</td><td>0.081</td><td>0.773</td><td>55.24</td><td>1.851</td><td>1.703</td></tr><tr><td>Lookout [47]</td><td>11.14</td><td>0.131</td><td>0.609</td><td>69.53</td><td>1.252</td><td>1.058</td><td>9.91</td><td>0.117</td><td>0.678</td><td>75.06</td><td>1.354</td><td>1.552</td></tr><tr><td>PhotoNVS [76]</td><td>12.02</td><td>0.147</td><td>0.558</td><td>48.03</td><td>1.404</td><td>1.306</td><td>10.83</td><td>0.132</td><td>0.609</td><td>52.51</td><td>1.708</td><td>1.602</td></tr><tr><td>GenWarp [55]</td><td>12.87</td><td>0.201</td><td>0.677</td><td>44.04</td><td>0.952</td><td>0.381</td><td>8.63</td><td>0.092</td><td>0.749</td><td>48.13</td><td>1.351</td><td>0.953</td></tr><tr><td>MotionCtrl [67]</td><td>13.34</td><td>0.192</td><td>0.698</td><td>43.11</td><td>0.863</td><td>0.724</td><td>8.12</td><td>0.087</td><td>0.779</td><td>47.54</td><td>1.452</td><td>1.161</td></tr><tr><td>CameraCtrl [16]</td><td>13.62</td><td>0.212</td><td>0.573</td><td>32.53</td><td>0.921</td><td>0.832</td><td>10.24</td><td>0.127</td><td>0.623</td><td>46.92</td><td>1.523</td><td>0.924</td></tr><tr><td>ViewCrafter [77]</td><td>16.17</td><td>0.226</td><td>0.598</td><td>31.02</td><td>1.304</td><td>0.953</td><td>8.97</td><td>0.112</td><td>0.649</td><td>45.23</td><td>1.651</td><td>1.052</td></tr><tr><td>SEVA [83]</td><td>16.63</td><td>0.331</td><td>0.469</td><td>31.04</td><td>1.203</td><td>0.851</td><td>12.16</td><td>0.181</td><td>0.508</td><td>36.03</td><td>1.422</td><td>0.954</td></tr><tr><td>VMem [32]</td><td>16.98</td><td>0.348</td><td>0.458</td><td>31.52</td><td>0.854</td><td>0.352</td><td>12.28</td><td>0.197</td><td>0.502</td><td>35.52</td><td>1.419</td><td>0.858</td></tr><tr><td>DFoT [57]</td><td>16.13</td><td>0.372</td><td>0.402</td><td>32.76</td><td>1.139</td><td>0.570</td><td>13.51</td><td>0.233</td><td>0.471</td><td>33.58</td><td>1.685</td><td>1.144</td></tr><tr><td>Ours</td><td>18.10</td><td>0.432</td><td>0.315</td><td>28.03</td><td>0.433</td><td>0.086</td><td>14.53</td><td>0.241</td><td>0.413</td><td>29.21</td><td>1.007</td><td>0.412</td></tr></table>

Table 3. Ablation studies on the RealEstate10K [84] dataset. We analyze the impact of our caching mechanism (top) and the spatialtemporal noise design (bottom).
<table><tr><td rowspan="2"></td><td colspan="6">Short-term  $( 5 0 ^ { t h } \mathrm { f r a m e } )$ </td><td colspan="6">Long-term  $( 2 0 0 ^ { t h } \mathrm { f r a m e } )$ </td></tr><tr><td>PSNR â</td><td>SSIMâ</td><td>LPIPS â</td><td>FID â</td><td> $R _ { \mathrm { d i s t } } \downarrow$ </td><td> $T _ { \mathrm { d i s t } } \downarrow$ </td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>FID â</td><td> $R _ { \mathrm { d i s t } } \downarrow$ </td><td> $T _ { \mathrm { d i s t } } \downarrow$ </td></tr><tr><td>No Cache</td><td>14.19</td><td>0.255</td><td>0.331</td><td>19.24</td><td></td><td></td><td>9.22</td><td>0.175</td><td>0.598</td><td>35.30</td><td></td><td></td></tr><tr><td>Caching by RGB point cloud</td><td>15.12</td><td>0.374</td><td>0.269</td><td>16.95</td><td>0.192</td><td>0.045</td><td>11.12</td><td>0.245</td><td>0.412</td><td>28.98</td><td>0.703</td><td>0.252</td></tr><tr><td>Caching by online optimized 3DGS</td><td>20.32</td><td>0.527</td><td>0.216</td><td>15.56</td><td>0.188</td><td>0.039</td><td>17.13</td><td>0.281</td><td>0.352</td><td>21.75</td><td>0.697</td><td>0.203</td></tr><tr><td>Full sequence noise</td><td>17.08</td><td>0.282</td><td>0.364</td><td>22.31</td><td>1.235</td><td>0.208</td><td>9.92</td><td>0.160</td><td>0.580</td><td>33.89</td><td>1.574</td><td>0.817</td></tr><tr><td>Spatial-varying noise</td><td>18.23</td><td>0.375</td><td>0.305</td><td>18.91</td><td>0.232</td><td>0.094</td><td>13.95</td><td>0.210</td><td>0.492</td><td>31.12</td><td>1.040</td><td>0.595</td></tr><tr><td>Temporal-varying noise</td><td>18.09</td><td>0.317</td><td>0.298</td><td>19.75</td><td>0.258</td><td>0.112</td><td>13.20</td><td>0.196</td><td>0.513</td><td>32.01</td><td>1.209</td><td>0.701</td></tr><tr><td>Spatial-temporal-varying noise</td><td>20.32</td><td>0.527</td><td>0.216</td><td>15.56</td><td>0.188</td><td>0.039</td><td>17.13</td><td>0.281</td><td>0.352</td><td>21.75</td><td>0.697</td><td>0.203</td></tr></table>

Noise Diffusion Model. The bottom half of the table validates our spatial-temporal noise diffusion design. Using a "Full sequence noise" (i.e., a standard video diffusion model) results in poor generation quality (9.92 long-term PSNR) and, critically, a catastrophic loss of camera control (1.574 $R _ { \mathrm { d i s t } } )$ . When using only "Spatial-varying noise," we observe a dramatic improvement in camera accuracy $( R _ { \mathrm { d i s t } }$ improves from 1.574 to 1.040), confirming that spatial noise is key for precise camera conditioning. Conversely, using only "Temporal-varying noise" improves generation quality (13.20 long-term PSNR) but fails to control the camera (1.209 $R _ { \mathrm { d i s t } } )$ . Our full "Spatial-temporal-varying noise"

model successfully combines both benefits, achieving the best generation quality (17.13 PSNR) and the best camera accuracy (0.697 $R _ { \mathrm { d i s t } } )$ , demonstrating the necessity and efficacy of our proposed noise diffusion strategy.

Table 4. Breakdown of latency and model size for each component in our pipeline. Times are in seconds (s).
<table><tr><td></td><td>VLM Prompting</td><td>Estimating 3D (TTT3R)</td><td>Optimizing 3DGS</td><td>Forward warping</td><td>ST-Diff 50 steps</td><td>Total</td></tr><tr><td>Inference time (s)</td><td>3.5</td><td>5.8</td><td>2.5</td><td>0.2</td><td>42.5</td><td>54.5</td></tr></table>

Inferencing efficiency. We provide a detailed breakdown of the inference latency per video chunk in Table 4. The average total time to generate one chunk (49 frames) is 54.5 seconds. The primary computational bottleneck is the iterative denoising process of our spatial-temporal diffusion model (ST-Diff), which requires 42.5 seconds for 50 steps, accounting for approximately 78% of the total time. In contrast, all 3D-related components are highly efficient: estimating the initial 3D representation with TTT3R takes 5.8s, optimizing the 3DGS cache takes only 2.5s, and forward warping is near-instant at 0.2s. This analysis demonstrates that the 3D-aware caching and conditioning, while critical for quality and consistency, add only a minimal computational overhead (8.5s total) compared to the main generative backbone.

## 6. Conclusion

In this work, we propose WorldWarp, a novel autoregressive framework for long-range, geometrically-consistent novel view extrapolation. Our method is designed to overcome the key limitation of prior work: the inability of standard generative models to handle imperfect 3D-warped priors. We introduce the ST-Diff model, a non-causal diffusion model trained with a spatially-temporally-varying noise schedule. This design explicitly trains the model to solve the fill-andrevise problem, simultaneously filling blank regions from pure noise while revising distorted content from a partiallynoised state. By coupling this model with an online 3D geometric cache to avoid irreversible error propagation, World-Warp achieves state-of-the-art performance, setting a new bar for long-range, camera-controlled video generation.

## References

[1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 5

[2] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5855â5864, 2021. 2

[3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased gridbased neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19697â 19705, 2023. 2

[4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2

[5] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023.

[6] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators, 2024. 2

[7] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. 2024. 2

[8] Boyuan Chen, Diego MartÃ­ MonsÃ³, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion

forcing: Next-token prediction meets full-sequence diffusion. In NeurIPS, 2024. 2

[9] Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. Ttt3r: 3d reconstruction as test-time training. arXiv preprint arXiv:2509.26645, 2025. 2, 3, 5, 6, 1

[10] Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. In ICLR, 2025. 2

[11] Rafail Fridman, Amit Abecasis, Yoni Kasten, and Tali Dekel. Scenescape: Text-driven consistent scene generation. Advances in Neural Information Processing Systems, 36:39897â 39914, 2023. 2, 3

[12] Kaifeng Gao, Jiaxin Shi, Hanwang Zhang, Chunping Wang, Jun Xiao, and Long Chen. Ca2-vdm: Efficient autoregressive video diffusion model with causal generation and cache sharing. arXiv preprint arXiv:2411.16375, 2024. 2

[13] Yuchao Gu, Weijia Mao, and Mike Zheng Shou. Long-context autoregressive video modeling with next-frame prediction. arXiv preprint arXiv:2503.19325, 2025. 2

[14] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Fei-Fei Li, Irfan Essa, Lu Jiang, and JosÃ© Lezama. Photorealistic video generation with diffusion models. In ECCV, 2024. 2

[15] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024. 2

[16] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 2, 3, 6, 7, 8

[17] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 6

[18] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey A. Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, and Tim Salimans. Imagen video: High definition video generation with diffusion models. ArXiv, abs/2210.02303, 2022. 2

[19] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. In NeurIPS, 2022. 2

[20] Lukas HÃ¶llein, Ang Cao, Andrew Owens, Justin Johnson, and Matthias NieÃner. Text2room: Extracting textured 3d meshes from 2d text-to-image models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7909â7920, 2023. 2, 3

[21] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 2

[22] Jinyi Hu, Shengding Hu, Yuxuan Song, Yufei Huang, Mingxuan Wang, Hao Zhou, Zhiyuan Liu, Wei-Ying Ma, and

Maosong Sun. Acdit: Interpolating autoregressive conditional modeling and diffusion transformer. arXiv preprint arXiv:2412.07720, 2024. 2

[23] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025. 2

[24] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. In ICLR, 2025. 2

[25] Bernhard Kerbl, Georgios Kopanas, Thomas LeimkÃ¼hler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023. 1, 2, 3, 5

[26] Jihwan Kim, Junoh Kang, Jinyoung Choi, and Bohyung Han. Fifo-diffusion: Generating infinite videos from text without training. In NeurIPS, 2024. 3

[27] Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jose Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. 2024. 2

[28] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2

[29] Xin Kong, Shikun Liu, Xiaoyang Lyu, Marwan Taher, Xiaojuan Qi, and Andrew J Davison. Eschernet: A generative model for scalable view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9503â9513, 2024. 2, 3

[30] Xin Kong, Daniel Watson, Yannick StrÃ¼mpler, Michael Niemeyer, and Federico Tombari. Causnvs: Autoregressive multi-view diffusion for flexible 3d novel view synthesis. arXiv preprint arXiv:2509.06579, 2025. 2

[31] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun Zhou, and Lin Gu. Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with global-local depth normalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20775â20785, 2024. 2

[32] Runjia Li, Philip Torr, Andrea Vedaldi, and Tomas Jakab. Vmem: Consistent interactive video scene generation with surfel-indexed view memory. arXiv preprint arXiv:2506.18903, 2025. 2, 3, 6, 7, 8

[33] Zhengqi Li, Qianqian Wang, Noah Snavely, and Angjoo Kanazawa. Infinitenature-zero: Learning perpetual view generation of natural scenes from single images. In European conference on computer vision, pages 515â534. Springer, 2022. 2, 7, 8

[34] Zongyi Li, Shujie Hu, Shujie Liu, Long Zhou, Jeongsoo Choi, Lingwei Meng, Xun Guo, Jinyu Li, Hefei Ling, and Furu Wei. Arlon: Boosting diffusion transformers with autoregressive models for long video generation. In ICLR, 2025. 2

[35] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024. 2

[36] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learningbased 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160â 22169, 2024. 6, 8, 1, 2

[37] Andrew Liu, Richard Tucker, Varun Jampani, Ameesh Makadia, Noah Snavely, and Angjoo Kanazawa. Infinite nature: Perpetual view generation of natural scenes from a single image. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14458â14467, 2021. 2, 7, 8

[38] Haozhe Liu, Shikun Liu, Zijian Zhou, Mengmeng Xu, Yanping Xie, Xiao Han, Juan C PÃ©rez, Ding Liu, Kumara Kahatapitiya, Menglin Jia, et al. Mardini: Masked autoregressive diffusion for video generation at scale. arXiv preprint arXiv:2410.20280, 2024. 2

[39] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9298â 9309, 2023. 2, 3

[40] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 2

[41] Norman MÃ¼ller, Katja Schwarz, Barbara RÃ¶ssle, Lorenzo Porzi, Samuel Rota Bulo, Matthias NieÃner, and Peter Kontschieder. Multidiff: Consistent novel view synthesis from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10258â10268, 2024. 2, 3

[42] Michael Niemeyer, Jonathan T Barron, Ben Mildenhall, Mehdi SM Sajjadi, Andreas Geiger, and Noha Radwan. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5480â5490, 2022. 2

[43] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024. 2

[44] Stefan Popov, Amit Raj, Michael Krainin, Yuanzhen Li, William T Freeman, and Michael Rubinstein. Camctrl3d: Single-image scene exploration with precise 3d camera control. In 2025 International Conference on 3D Vision (3DV), pages 649â658. IEEE, 2025. 2, 3

[45] Shuhuai Ren, Shuming Ma, Xu Sun, and Furu Wei. Next block prediction: Video generation via semi-auto-regressive modeling. arXiv preprint arXiv:2502.07737, 2025. 2

[46] Weining Ren, Zihan Zhu, Boyang Sun, Jiaqi Chen, Marc Pollefeys, and Songyou Peng. Nerf on-the-go: Exploiting uncertainty for distractor-free nerfs in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8931â8940, 2024. 2

[47] Xuanchi Ren and Xiaolong Wang. Look outside the room: Synthesizing a consistent long-term 3d scene video from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3563â3573, 2022. 6, 7, 8

[48] Robin Rombach, Patrick Esser, and BjÃ¶rn Ommer. Geometryfree view synthesis: Transformers and no 3d priors. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14356â14366, 2021. 2, 7, 8

[49] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and BjÃ¶rn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684â10695, 2022. 2, 3

[50] David Ruhe, Jonathan Heek, Tim Salimans, and Emiel Hoogeboom. Rolling diffusion models. 2024. 3

[51] Sara Sabour, Suhani Vora, Daniel Duckworth, Ivan Krasin, David J Fleet, and Andrea Tagliasacchi. Robustnerf: Ignoring distractors with robust losses. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20626â20636, 2023. 2

[52] Sara Sabour, Lily Goli, George Kopanas, Mark Matthews, Dmitry Lagun, Leonidas Guibas, Alec Jacobson, David Fleet, and Andrea Tagliasacchi. Spotlesssplats: Ignoring distractors in 3d gaussian splatting. ACM Transactions on Graphics, 44 (2):1â11, 2025. 2

[53] Sand-AI. Magi-1: Autoregressive video generation at scale, 2025. 3

[54] Kyle Sargent, Zizhang Li, Tanmay Shah, Charles Herrmann, Hong-Xing Yu, Yunzhi Zhang, Eric Ryan Chan, Dmitry Lagun, Li Fei-Fei, Deqing Sun, et al. Zeronvs: Zero-shot 360- degree view synthesis from a single real image. 2023. 2, 3

[55] Junyoung Seo, Kazumi Fukuda, Takashi Shibuya, Takuya Narihira, Naoki Murata, Shoukang Hu, Chieh-Hsin Lai, Seungryong Kim, and Yuki Mitsufuji. Genwarp: Single image to novel views with semantic-preserving generative warping. Advances in Neural Information Processing Systems, 37: 80220â80243, 2024. 2, 3, 6, 7, 8

[56] Vincent Sitzmann, Semon Rezchikov, Bill Freeman, Josh Tenenbaum, and Fredo Durand. Light field networks: Neural scene representations with single-evaluation rendering. Advances in Neural Information Processing Systems, 34:19313â 19325, 2021. 3

[57] Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du, Russ Tedrake, and Vincent Sitzmann. History-guided video diffusion. arXiv preprint arXiv:2502.06764, 2025. 3, 7, 8

[58] Mingzhen Sun, Weining Wang, Gen Li, Jiawei Liu, Jiahui Sun, Wanquan Feng, Shanshan Lao, SiYu Zhou, Qian He, and Jing Liu. Ar-diffusion: Asynchronous video generation with auto-regressive diffusion. In CVPR, 2025. 3

[59] R Villegas, H Moraldo, S Castro, M Babaeizadeh, H Zhang, J Kunze, PJ Kindermans, MT Saffar, and D Erhan. Phenaki: Variable length video generation from open domain textual descriptions. In ICLR, 2023. 2

[60] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao

Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 3, 4, 6, 1

[61] Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2

[62] Guangcong Wang, Zhaoxi Chen, Chen Change Loy, and Ziwei Liu. Sparsenerf: Distilling depth ranking for few-shot novel view synthesis. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9065â9076, 2023. 2

[63] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294â5306, 2025. 2

[64] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697â20709, 2024. 6

[65] Yuqing Wang, Tianwei Xiong, Daquan Zhou, Zhijie Lin, Yang Zhao, Bingyi Kang, Jiashi Feng, and Xihui Liu. Loong: Generating minute-level long videos with autoregressive language models. arXiv preprint arXiv:2410.02757, 2024. 2

[66] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600â612, 2004. 6

[67] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1â11, 2024. 2, 3, 6, 7, 8

[68] Dirk Weissenborn, Oscar TÃ¤ckstrÃ¶m, and Jakob Uszkoreit. Scaling autoregressive video models. In ICLR, 2020. 2

[69] Wenming Weng, Ruoyu Feng, Yanhui Wang, Qi Dai, Chunyu Wang, Dacheng Yin, Zhiyuan Zhao, Kai Qiu, Jianmin Bao, Yuhui Yuan, et al. Art-v: Auto-regressive text-to-video generation with diffusion models. In CVPR, 2024. 2

[70] Desai Xie, Zhan Xu, Yicong Hong, Hao Tan, Difan Liu, Feng Liu, Arie Kaufman, and Yang Zhou. Progressive autoregressive video diffusion models. arXiv preprint arXiv:2410.08151, 2024. 3

[71] Jiacong Xu, Yiqun Mei, and Vishal Patel. Wild-gs: Realtime novel view synthesis from unconstrained photo collections. Advances in Neural Information Processing Systems, 37:103334â103355, 2024. 2

[72] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021. 2

[73] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2

[74] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In ICLR, 2025. 2

[75] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In CVPR, 2025. 2

[76] Jason J Yu, Fereshteh Forghani, Konstantinos G Derpanis, and Marcus A Brubaker. Long-term photometric consistent novel view synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7094â7104, 2023. 2, 7, 8

[77] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024. 2, 3, 7, 8

[78] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19447â19456, 2024. 2

[79] Dongbin Zhang, Chuming Wang, Weitao Wang, Peihao Li, Minghan Qin, and Haoqian Wang. Gaussian in the wild: 3d gaussian splatting for unconstrained image collections. In European Conference on Computer Vision, pages 341â359. Springer, 2024. 2

[80] Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction models for video generation. arXiv preprint arXiv:2504.12626, 2025. 3

[81] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586â595, 2018. 6

[82] Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T Freeman, and Hao Tan. Test-time training done right. arXiv preprint arXiv:2505.23884, 2025. 2

[83] Jensen Zhou, Hang Gao, Vikram Voleti, Aaryaman Vasishta, Chun-Han Yao, Mark Boss, Philip Torr, Christian Rupprecht, and Varun Jampani. Stable virtual camera: Generative view synthesis with diffusion models. arXiv preprint arXiv:2503.14489, 2025. 2, 7, 8

[84] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817, 2018. 6, 7, 8, 1, 2

[85] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang. Fsgs: Real-time few-shot view synthesis using gaussian splatting. In European conference on computer vision, pages 145â163. Springer, 2024. 2

# WorldWarp: Propagating 3D Geometry with Asynchronous Video Diffusion

Supplementary Material

## 7. Implementation Details

Training. We implement our WorldWarp using the Wan2.1- T2V-1.3B diffusion transformer [60] as the generative backbone. All parameters of the model are fully fine-tuned in an end-to-end manner. The model is trained for 10k steps with a global batch size of 64 (8 per GPU) on 8 NVIDIA H200 GPUs. We utilize the AdamW optimizer with a learning rate of $5 \times 1 0 ^ { - 5 }$ and apply a 1,000-step linear warmup. The training video resolution is set $\mathrm { t o 4 8 0 \times 7 2 0 }$

Inference. The video generation process is initiated from a single source image. Subsequent content is synthesized auto-regressively in chunks of 49 frames. To ensure temporal continuity, we utilize a fixed context overlap of 5 frames for every iteration following the initial chunk. To establish the global coordinate system, we first estimate camera poses and intrinsics from the reference video using TTT3R [9]. For generation lengths exceeding the reference trajectory, we employ a velocity-based extrapolation strategy, computing linear velocity for translation and Spherical Linear Interpolation (SLERP) for rotation based on a 20-frame smoothing window. During the generation of each chunk, we optimize the online 3DGS cache for 500 iterations with a learning rate of $1 . 6 \times 1 0 ^ { - 3 }$ to render the warped priors $\mathcal { X } _ { s  \nu }$

We utilize the spatially-temporally schedule described in Fig. 3 in the main text for the reverse diffusion process. Specifically, the latent representations of the 5 context overlap frames are enforced as hard constraints. For the target frames, we set the strength parameter $\tau = 0 . 8$ . Consequently, regions with valid geometric warps $( \mathbf { M } _ { \mathrm { l a t e n t } } = 1 )$ are initialized with a reduced noise level $\sigma _ { \mathrm { s t a r t } }$ corresponding to Ï , preserving the structural integrity of the 3D cache. Conversely, occluded or blank regions $( \mathbf { M } _ { \mathrm { l a t e n t } } = 0 )$ are initialized with standard Gaussian noise $( \sigma _ { \mathrm { f i l l e d } } = \sigma _ { T _ { N } } \approx 1 . 0 )$ to facilitate generative inpainting. The diffusion model employs a Flow Match Euler Discrete Scheduler with 50 denoising steps.

## 8. More Experiment Results

## 8.1. Visualization Results

We illustrate more results on the RealEstate10K [84] and DL3DV [36] datasets in Fig. 6 and Fig. 7. Please refer to video supplementary material for more results.

To further demonstrate the robust generalization capacity of ST-Diff, we extend our evaluation beyond standard photorealistic benchmarks to scenes rendered in diverse artistic styles in Fig. 8. By prompting the model with specific stylistic descriptors, such as "Van Gogh style" or "Studio Ghibli style," we generate a variety of stylized video sequences.

The results illustrate that our method successfully synthesizes these highly stylized scenes while strictly preserving the underlying 3D geometric consistency. These qualitative results validate that our proposed training strategy effectively integrates fine-grained geometric control without sacrificing the rich semantic and aesthetic generalization capabilities inherent in the pre-trained model. This confirms that adapting the foundation model into an asynchronous diffusion framework does not compromise its ability to interpret opendomain text prompts.

## 8.2. Analysis of Spatially-Adaptive Noise Dynamics

To validate the effectiveness of our geometry-aware inference strategy, we visualize the evolution of the noise schedule matrix $\Sigma _ { \mathcal { V } }$ throughout the reverse diffusion process. As shown in Fig. 9, the visualization is structured as a spatiotemporal grid where each row corresponds to a latent temporal token t (derived from the 49 video frames via VAE encoding) and columns progress through the denoising steps from left $( T = 9 9 9 )$ to right $( T = 0 )$

The map explicitly corroborates our dual-schedule formulation. The top two rows, representing the 5 history context frames (corresponding $\mathbf { t o } \sim 2$ latent tokens), remain fully constrained with zero noise (dark purple) throughout the process, ensuring seamless transitions from previous chunks. In the subsequent 11 rows (the generated tokens), we observe a distinct spatial modulation. The valid geometric regions, projected from the 3DGS cache, are maintained at a reduced noise level Ï (intermediate green/teal) to preserve high-fidelity structural details. In contrast, occluded or blank regions are initialized with high-variance noise (yellow) to facilitate the generative hallucination of new content. This confirms that our model effectively balances geometric preservation with generative inpainting during the autoregressive process.

## 9. Limitations

Despite the effectiveness of ST-Diff in generating geometrically consistent long-term videos, our method is subject to certain limitations common to autoregressive video generation frameworks.

Error Accumulation in Long-horizon Generation. Although our model is trained in an asynchronous diffusion manner, where we apply varying noise strengths to different frames and spatial regions to mimic inference conditions, generating infinite-length video sequences with perfect fidelity remains an unresolved challenge. In our autoregressive

<!-- image-->  
1 st frame

<!-- image-->  
50th frame

<!-- image-->  
100th frame

<!-- image-->  
150 th frame

<!-- image-->  
200th frame

<!-- image-->  
225th frame

Figure 6. Qualitative results on the RealEstate10K [84] datasets.  
<!-- image-->  
1 st frame

<!-- image-->  
50th frame

<!-- image-->  
100th frame

<!-- image-->  
150 th frame

<!-- image-->  
200th frame

<!-- image-->  
225th frame

Figure 7. Qualitative results on the DL3DV [36] datasets.

pipeline, the generated output of one chunk serves as the historical context for the next. Consequently, minor visual artifacts or geometric inconsistencies can propagate and accumulate over time. For extremely long sequences, such as those exceeding 1000 frames, this drift can eventually lead to degradation in visual quality or geometric stability. This remains a persistent issue shared by state-of-the-art video generation methods.

Dependency on Geometric Priors. Our method operates on the premise that forward-warped images provide strong geometric hints for generation. Therefore, our performance is heavily dependent on the accuracy of the upstream 3D geometry foundation models, such as TTT3R [9] or VGGT [63], used for depth and camera pose estimation. In scenarios where these pre-trained estimators struggle, including complex outdoor environments with extreme lighting, transparency, or lack of texture, the estimated depth maps and poses may be inaccurate. This results in incorrect warping results that deviate significantly from the true geometry. While ST-Diff is designed to correct artifacts, it may fail to recover high-quality frames when the geometric guidance is fundamentally flawed or contains excessive noise.

<!-- image-->  
Figure 8. Generalization to Out-of-Distribution Artistic Styles. We evaluate the robustness of ST-Diff by generating video sequences conditioned on diverse artistic text prompts (e.g., âVan Gogh styleâ, âStudio Ghibli styleâ). The visualized chunks demonstrate that our method successfully synthesizes high-quality stylized content while maintaining rigorous 3D geometric consistency across the autoregressive generation. This confirms that our geometry-aware fine-tuning strategy effectively incorporates structural control without compromising the semantic and aesthetic generalization capabilities of the pre-trained diffusion backbone.

<!-- image-->  
Figure 9. Visualization of the Spatially-Adaptive Noise Schedule. We visualize the schedule matrix $\Sigma _ { \mathcal { V } }$ across the reverse diffusion process. The horizontal axis represents the progression of denoising steps (from $T = 9 9 9 \mathrm { t o } 0 )$ , while the vertical axis corresponds to the sequence of temporal tokens (13 tokens derived from 49 frames). The top two rows represent the History Tokens (context), which are enforced as hard constraints (dark purple, $\sigma = 0 )$ . The subsequent rows represent the Generated Tokens, where the noise levels are spatially modulated: (1) Valid warped regions are held at a reduced noise level Ï (intermediate green) to preserve explicit geometry; and (2) Occluded regions are initialized with maximal noise (yellow, $\sigma \approx 1 . 0 )$ to enable the generative synthesis of novel content.