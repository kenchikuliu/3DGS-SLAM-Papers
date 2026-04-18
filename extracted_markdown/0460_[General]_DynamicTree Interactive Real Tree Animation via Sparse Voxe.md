# DynamicTree: Interactive Real Tree Animation via Sparse Voxel Spectrum

Yaokun Li1,2 Lihe Ding2 Xiao Chen2 Guang Tan1â Tianfan Xue2,3\* 1Sun Yat-sen University 2CUHK MMLab 3CPII under InnoHK

<!-- image-->  
Figure 1. DynamicTree achieves structurally consistent and realistic long-term animation and dynamic interaction on real-world 3DGS trees. It first generates mesh motion via a compact sparse voxel spectrum representation and then deforms the surface-bound Gaussian primitives. We visualize the slice of the generated motion at the orange scanline along the time dimension.

## Abstract

Generating dynamic and interactive 3D trees has wide applications in virtual reality, games, and world simulation. However, existing methods still face various challenges in generating structurally consistent and realistic 4D motion for complex real trees. In this paper, we propose DynamicTree, the first framework that can generate long-term, interactive 3D motion for 3DGS reconstructions of real trees. Unlike prior optimization-based methods, our approach generates dynamics in a fast feed-forward manner. The key success of our approach is the use of a compact sparse voxel spectrum to represent the tree movement. Given a 3D tree from Gaussian Splatting reconstruction, our pipeline first generates mesh motion using the sparse voxel spectrum and then binds Gaussians to deform the mesh. Additionally, the proposed sparse voxel spectrum can also serve as a basis for fast modal analysis under external forces, allowing real-time interactive responses. To train our model, we also introduce 4DTree, the first large-scale synthetic 4D tree dataset containing 8,786 animated tree meshes with 100-frame mo-

tion sequences. Extensive experiments demonstrate that our method achieves realistic and responsive tree animations, significantly outperforming existing approaches in both visual quality and computational efficiency. Intuitive video results are best viewed on our project webpage https://dynamictree-dev.github.io/DynamicTree.github.io/.

## 1. Introduction

With recent advances of neural radiance fields (NeRF) [28] and 3D Gaussian Splatting (3DGS) [20], high-quality reconstruction and real-time rendering become feasible. Driven by these, there is high demand to make static reconstructions interactable, for immersive experiences like 3D games, movies, and virtual reality [11, 19, 36]. As a vital component of natural landscapes, tree animation can significantly enrich immersive digital experiences. For example, when viewing a reconstructed backyard on a VR headset, if trees can sway gently in the wind or respond to dragging interaction, it would significantly enhance immersion.

However, animating a realistic 3D tree remains challenging. Traditional tree animation methods [30â32] typically construct structurally approximated physical tree models and then perform simulation. Although such approaches can simulate realistic motion dynamics, they are either confined to synthetic tree models or require labor-intensive mesh refinements for high-quality rendering. Consequently, they struggle to reproduce the immersive realism and structural complexity of real-world trees, unlike animations based on reconstructed 3DGS representations.

To animate 3DGS trees, existing methods can be broadly categorized into 4D generation and physics-based simulation. 4D generation approaches [3, 24, 27, 38, 50, 52, 56] typically use 2D motion priors from video diffusion models (VDMs) to optimize 4D representations. However, such approaches tend to generate simple motions with temporal and spatial inconsistencies, making realistic tree animation challenging and demanding heavy per-scene optimization. Another category, physics-based approaches, such as Phys-Gaussian [48], couple 3DGS with physical simulation engines like the Material Point Method (MPM) [18], achieving better 3D consistency. However, they rely on simplified constitutive models and approximate boundary conditions, making fine-grained, realistic branch- and leaf-level motion difficult. These solvers are also computationally expensive, making real-time applications impractical.

In this work, we introduce DynamicTree, a novel 4D generative framework for animating 3D Gaussian Splatting (3DGS) trees with real-time interactivity. Unlike VDMbased 4D generation methods, DynamicTree directly learns tree motion priors in 3D space, avoiding geometric inconsistencies and capturing fine-grained, physically plausible dynamics. Compared with computationally expensive physics-based optimization, our generative model synthesizes fine-grained 3D tree motion in a fast feed-forward manner, achieving more than a hundred times acceleration. To train the model, we construct the first large-scale 4D tree motion dataset containing 8,786 animated tree meshes with 100-frame sequences, generated via hierarchical branching simulation [45].

Still, even with this dataset, direct motion prediction in 3D space is challenging, as it requires both an efficient and robust representation of 3D motion. Since a reconstructed 3DGS tree typically contains hundreds of thousands of Gaussians, naive long-term motion prediction is prohibitively expensive in memory and training data. Thus, an effective animation strategy is needed to reduce computational and data costs. Furthermore, since the training data are synthetic while testing targets real reconstructed trees, a robust 3D representation is essential to bridge the syntheticto-real gap.

Our framework addresses these challenges with a twostage pipeline. We first generate mesh motion and then bind Gaussians to the deforming mesh, allowing full 3DGS deformation while only modeling mesh dynamics [12, 43].

To further improve efficiency and generalization, we introduce a sparse voxel-based motion representation that both reduces the complexity of dense vertex deformations and mitigates the synthetic-to-real gap by converting irregular mesh sampling into a unified voxel structure. Moreover, inspired by previous work [23], we further model the motion of each voxel as a spectrum, representing long-term mesh dynamics with only a few frequency components, further reducing temporal complexity. At last, we can treat the predicted 3D spectrum as 3D modal bases [23] for modal analysis and approximate 3D interactions as a linear combination of base motions. By doing so, it reduces the interaction simulation to about 18ms, making it significantly faster than MPM-based simulation and enabling real-time interaction.

We evaluate our method through comparative experiments on various real-world trees, showing that our approach produces more natural animations of trees swaying in the wind and dynamically interacting. We summarize our contributions as follows:

â¢ We introduce DynamicTree, a novel framework for 3D motion generation of real-world trees, enabling physically plausible and structurally consistent animations.

â¢ We propose a novel sparse voxel spectrum representation for efficient long-term 4D tree motion generation and fast interactive simulation under external forces.

â¢ To support complex 3D tree motion generation, we contribute 4DTree, a large-scale synthetic 4D tree dataset containing 8,786 animated tree meshes, each with 100- frame motion sequences.

## 2. Related Work

## 2.1. Tree Animation

Traditional methods for tree animation typically rely on constructing physical tree models and simulating their dynamics. Early works represent trees as articulated rigid bodies or particle-based systems connected by springs or links [2, 32], enabling physically plausible yet simplified deformations. More advanced approaches, such as Windy-Tree [30], couple growth models with fluid dynamics to simulate wind effects, while others introduce semiautomatic pipelines for interactive wind and drag simulations [55]. However, these methods often depend on structurally approximated synthetic physical trees and require extensive manual preparation to make them simulationready. Consequently, they struggle to faithfully reproduce the intricate geometry and motion of real-world trees and frequently exhibit suboptimal rendering quality compared with real-world 3DGS-based tree animations.

## 2.2. 4D Generation

Recently, 4D content generation has gained growing attention in generative AI. These methods typically optimize 3D motion from 2D diffusion priors. Works such as MAV3D [38], Dream-in-4D [56], and CT4D [47] use SDS-based pipelines to generate and animate 3D content, while Comp4D [1] and 4Dynamic [53] leverage language or video priors for motion synthesis. DreamGaussian4D [34], EG4D [41], 4DGen [52], and SV4D [49, 51] further integrate image-to-3D, video, or multi-view diffusion models [25, 42] for 4D generation. Despite recent advances, these methods depend on 2D motion priors from video diffusion models, leading to temporal and spatial inconsistencies and noticeable artifacts, especially when generating the complex motions of trees. In addition, their reliance on scene-specific optimization incurs substantial computational cost. In contrast, our method learns a 3D diffusion prior from 4D tree data to predict consistent motion in a fast feed-forward manner.

## 2.3. Physics-based 3DGS Simulation

Physics-based dynamic generation methods use the differentiable MPM simulation framework to optimize 3DGS dynamics. The pioneering PhysGaussian [48] employs a customized MPM formulation that bridges Newtonian dynamics and 3D Gaussian kernels, enabling the simulation of material behaviors. To reduce manual parameter tuning, recent works combine MPM with VDMs or LLMs to estimate physical properties. For instance, PhysDreamer [54] and Dreamphysics [16], for example, integrate VDM motion priors with MPM to learn dynamic properties such as Youngâs modulus and Poissonâs ratio. PhysFlow [26] initializes parameters via GPT-4 [1] and further optimizes them using optical flow guidance. However, as precisely setting individual parameters for each part is challenging, these methods often assume uniform material properties across the entire object to simplify optimization. This facilitates global motion coherence but suppresses local deformation details and reduces visual realism in tree animation. Additionally, the high computational cost of MPM-based simulation limits its use in real-time applications.

## 2.4. Spectrum-based Motion Representation

Quasi-periodic motions of plants and trees are well-suited for spectrum-based modeling. Prior work shows they can be modeled as a superposition of a few harmonic oscillators at different frequencies [7, 9, 10]. Generative-Dynamics [23] leverages this property by reconstructing long videos from a few generated frequency components. Moreover, Abe et al. [8] demonstrate that spectral volumes can serve as image-space modal bases for plausible interactive simulation via modal analysis. Building on this idea, Generative-Dynamics and ModalNeRF [29] apply similar principles to image-space and implicit NeRF representation, achieving interactive dynamic simulations. Inspired by these works, we propose a sparse voxel spectrum representation that enables efficient long-term motion generation and interactive simulation for 3D trees.

## 3. methodology

## 3.1. Task Formulation

Given multi-view images of a static tree, our goal is to generate a 4D model as a deformed 3DGS sequence $\mathcal { G } =$ $\bar { \{ G _ { t } | G _ { t } = \{ x _ { i } ^ { t } , r _ { i } ^ { t } , s _ { i } ^ { t } , \sigma _ { i } ^ { t } , c _ { i } ^ { t } \} _ { i = 1 } ^ { H } \} _ { t = 0 } ^ { T } }$ , where $x _ { i } ^ { t } , r _ { i } ^ { t } , s _ { i } ^ { t } , \sigma _ { i } ^ { t }$ and $c _ { i } ^ { t }$ denote the position, rotation, scale, opacity, and color of the i-th Gaussian primitive at frame t, respectively. This requires predicting temporal deformations of the static 3DGS: $\bar { \mathcal { D } } _ { g } = \{ D _ { g } ^ { t } | D _ { g } ^ { \bar { t } } = ( \Delta x _ { i } ^ { t } \in \mathbb { R } ^ { 3 } , \Delta r _ { i } ^ { t } \in \mathbb { R } ^ { 4 } , \Delta s _ { i } ^ { t } \in$ $\mathbb { R } ^ { 3 } ) \} _ { i = 1 , t = 1 } ^ { H , T }$ . Previous methods [26, 52] typically rely on optimization-based strategies to solve this problem, which are computationally expensive. We instead formulate this task as a conditional generation problem. To handle largescale primitives efficiently, we propose a two-stage pipeline, named DynamicTree, as shown in Fig. 2. First, we introduce the sparse voxel spectrum (Â§3.2) representation to efficiently represent the motion. Then, we extract voxel grid conditions (Â§3.3) and employ a sparse voxel diffusion module (Â§3.4) to generate mesh motion. Subsequently, a twostage optimization strategy is proposed in Â§3.5 to refine performance. Finally, we bind 3DGS on the animated mesh surface (Â§3.6) to compute $\mathcal { D } _ { g }$

## 3.2. Sparse Voxel Spectrum

The motion of a mesh sequence can be represented as $\mathcal { D } _ { m } =$ $\{ D _ { m } ^ { t } \ \in \ \mathbb { R } ^ { 3 \times N } | t \ = \ 1 , . . . , T \}$ , where $D _ { m } ^ { t } ( i )$ denotes the displacement vector of the i-th vertex relative to its initial frame position at time t. Although simpler than predicting the 3DGS deformation $\mathcal { D } _ { g } = \{ D _ { q } ^ { t } \in \mathbb { R } ^ { 1 0 \times H } | t = 1 , . . . , T \}$ , where $D _ { g } ^ { t }$ is the deformation of center, scale, and quaternion rotation for a Gaussian blob, mesh motion remains challenging due to the large number of vertices N and frames $T .$

Prior works [21, 44] exploit the spatial sparsity of 3D motion using compact bases (e.g., [44] drives 40k Gaussians with only 20 motion bases). In our case, tree-like motions also exhibit such sparsity. For example, vertices within the same leaf or local branch tend to show similar motion patterns. However, relying solely on sparse motion bases like [44] will make it difficult to model finegrained details due to the complexity of tree motion. Therefore, we propose representing tree motion using sparse voxels, where all vertices in a voxel share the same displacement. With this, to predict dense mesh motion, we only need to predict sparse voxel motion $\mathcal { D } _ { v } = \{ D _ { v } ^ { t } \in \mathbb { R } ^ { 3 \times n } | t =$ $1 , . . . , T \}$ }, where n is typically an order of magnitude smaller than N .

To ensure temporal consistency over long sequences, instead of autoregressive or time-conditioned generation [4, 5,

<!-- image-->  
Figure 2. Our framework animates 3DGS trees in two stages: (1) spectrum-based motion generation in the frequency domain, and (2) deformation transfer to 3DGS through mesh binding. In the first stage, we extract the tree mesh from multi-view images, voxelize it, and encode it into a sparse voxel latent condition. A sparse voxel diffusion model then generates a compact motion representation $s ,$ which is used to reconstruct mesh motion via devoxelization and inverse Fast Fourier Transform. In the second stage, 3DGS primitives are bound to the mesh surface and animated by its deformations.

22], we draw inspiration from Generative-Dynamics [23], which models motion via low-frequency components of spectral volumes [8]. Inspired by this, we introduce the sparse voxel spectrums to represent 3D motion. Specifically, for sparse voxel motion $\mathcal { D } _ { v } \ \in \ \mathbb { R } ^ { 3 \times n \times T }$ , we apply the Fast Fourier Transform (FFT) along the temporal dimension, resulting in a complex-valued frequency-domain representation $\hat { \mathcal { D } } _ { v } \in \mathbb { C } ^ { 3 \times n \times \hat { T } }$ , where each spatial displacement is decomposed into its corresponding components. Tree-like quasi-periodic motions are predominantly captured by the first K frequency components. Thus, a compact representation $\hat { \mathcal { D } } _ { v } ^ { ( K ) } \in \mathbb { C } ^ { \mathrm { 3 } \times n \times \bar { K } }$ is sufficient for nearly lossless reconstruction of the full spectrum $\hat { \mathcal { D } } _ { v } \in \mathbb { C } ^ { 3 \times n \times \check { T } }$ with K = 16 following [23]. Then, to facilitate the generation of these top-K frequency components for the sparse voxel motion, we introduce the sparse voxel spectrum representation $\mathcal { S } = \{ s _ { i } \in \mathbb { R } ^ { 6 \times n } | i = 1 , . . . , K \}$ , where the dimension of size 6 corresponds to the real and imaginary parts of the $x , y ,$ and z dimensions. Given this representation, we can reconstruct the mesh motion through the following operation:

$$
\mathcal { D } _ { m } = \mathrm { D e v } ( \mathrm { i F F T } ( S ) )\tag{1}
$$

where iFFT denotes the inverse FFT along the temporal dimension, and Dev(Â·) represents the devoxelization process that maps sparse voxel displacements to dense mesh vertex motion.

## 3.3. Voxel Grid Condition

To reduce the synthetic-to-real gap when using multi-view images as input, we condition the motion generation model on voxel grids. Given multi-view images of a static tree, we first reconstruct its mesh $M = ( V , F )$ using an off-theshelf method [13], where $V = \{ v _ { i } \in \mathbb { R } ^ { 3 } \} _ { i = 1 } ^ { N }$ and $F =$ $\{ f _ { j } \subset \{ 1 , \dots , N \} \} _ { j = 1 } ^ { P }$ denote the sets of vertices and faces, respectively. We then voxelize the mesh to obtain a sparse voxel grid G, which serves as the conditioning input.

## 3.4. Sparse Voxel Diffusion

Before performing the diffusion generation, the sparse voxel grid G is encoded via a sparse encoder with several sparse convolutional blocks [46], resulting in a compact latent representation $g ~ \in ~ \mathbb { R } ^ { d \times n }$ as geometric conditioning. Our sparse voxel diffusion module builds on the U-Net architecture introduced by XCube [35]. Specifically, to generate the sparse voxel spectrum $\boldsymbol { \mathcal { S } } = \{ \boldsymbol { s } _ { i } \in \mathbb { R } ^ { 6 \times n } \ | \ i =$ $1 , \ldots , K \}$ of the mesh motion, we condition the diffusion generation process on both the frequency index and the latent feature g, generating each frequency component separately. The diffusion process [15] starts from pure Gaussian noise and iteratively predicts noise over L Markov steps. At each step, we concatenate the latent feature g with the noisy latent $s _ { l } ,$ and inject the frequency embedding into every ResBlock of the sparse voxel U-Net through scale and shift operations.

## 3.5. Optimization

We directly supervise the sparse voxel spectrums during training. To achieve this, we construct a 4D dataset of tree mesh motion sequences, which is detailed in Sec. 5. Given the constructed mesh motion, we first apply the FFT to obtain the spectrum for each vertex. Then, we voxelize the mesh motion spectrum to create the ground truth of sparse voxel spectrums.

During training, we diffuse the sparse voxel spectrum at each frequency component over L diffusion steps and supervise the modelâs prediction using the following diffusion loss:

$$
\begin{array} { r } { \mathcal { L } _ { D M } = \mathbb { E } _ { \epsilon \sim \mathcal { N } ( 0 , I ) , l \sim \mathcal { U } ( \{ 1 , \dots , L \} ) } \left[ \Vert \epsilon - \epsilon _ { \theta } ( \mathbf { s } _ { l } ; l , g , f ) \Vert ^ { 2 } \right] , } \end{array}\tag{2}
$$

where g and $f$ denote the sparse voxel latent condition and frequency embedding, respectively. Further, we find that using only the diffusion loss $\mathcal { L } _ { D M }$ may lead to unrealistic motion, such as divergence of some branches, as shown in Fig. 5, because the problem is under-constrained. To address this, we introduce a Local Spectrum Smoothness (LSS) loss that encourages local consistency in the frequency domain, inspired by the physical prior proposed by Sorkine and Alexa [40] that neighboring points tend to move similarly. Specifically, we compute discrepancies in the frequency-domain parameters between each point and its neighbors, weighted by spatial proximity:

$$
\mathcal { L } _ { \mathrm { L S S } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { j \in \mathcal { N } ( i ) } e ^ { - \alpha d _ { i j } } \left( \Vert \mathbf { R e } _ { i } - \mathbf { R e } _ { j } \Vert + \lambda \Vert \mathrm { I m } _ { i } - \mathrm { I m } _ { j } \Vert \right)\tag{3}
$$

where $ { \mathrm { R e } } _ { i }$ and $\operatorname { I m } _ { i }$ denote the real and imaginary components of the spectrum at point $i , \mathcal { N } ( i )$ represents its Îº- nearest neighbors, $d _ { i j }$ denotes the Euclidean distance between points i and $j ,$ , and Î» controls the weight of the imaginary part.

Moreover, we observe that naively combining both losses $\mathcal { L } _ { D M }$ and $\mathcal { L } _ { L S S }$ from the beginning of training would also lead to unstable learning. To address this issue, we adopt a two-stage training strategy: in the first stage, we train the model using only $\mathcal { L } _ { D M }$ for a certain number of iterations; in the second stage, we introduce $\mathcal { L } _ { L S S }$ to refine the spectral representation. This strategy significantly improves training stability and overall performance.

## 3.6. Mesh-Driven 3DGS Animation

With the generated sparse voxel spectrums of a given mesh M, we need to decode it to a full motion field of 3DGS. To do that, we first devoxelize the sparse spectrum by assigning the same spectrum to all vertices within the same voxel. Then, we convert spectrums to the time-domain mesh motion $\mathcal { D } _ { m }$ through the inverse Fast Fourier Transform.

With the recovered mesh motion, we then animate the 3DGS model, by binding Gaussian primitives to the mesh surface, as proposed by GaMeS [43]. This operation can be viewed as a reparameterization: for each face $f _ { j } ~ =$ $\{ v _ { 1 } , v _ { 2 } , v _ { 3 } \} \in \mathbb { R } ^ { 3 }$ , we parameterize the attributes of its associated Gaussian primitive $( u , r ,$ , and s) using the positions

of the three vertices:

$$
\left\{ \begin{array} { l } { { \mu = \alpha _ { 1 } V _ { 1 } + \alpha _ { 2 } V _ { 2 } + \alpha _ { 3 } V _ { 3 } , } } \\ { { r = [ r _ { 1 } ( f _ { i } ) , ~ r _ { 2 } ( f _ { i } ) , ~ r _ { 3 } ( f _ { i } ) ] , } } \\ { { s = \mathrm { d i a g } ( s _ { 1 } ( f _ { i } ) , ~ s _ { 2 } ( f _ { i } ) , ~ s _ { 3 } ( f _ { i } ) ) , } } \end{array} \right.\tag{4}
$$

where $\alpha _ { 1 } , \alpha _ { 2 } .$ , and $\alpha _ { 3 }$ are learnable parameters, and $r _ { 1 } , r _ { 2 } ,$ $r _ { 3 } , s _ { 1 } , s _ { 2 } .$ , and $s _ { 3 }$ are parameterization functions. For details, please refer to [43]. Through this binding strategy, we can compute the 3DGS deformation $\mathcal { D } _ { g }$ directly from the mesh motion $\mathcal { D } _ { m }$ . This allows us to obtain the final deformed 3DGS sequence $\mathcal { G } .$

## 4. Interactive Simulation with Modal Analysis

Modal analysis is a technique used to decompose complex deformable motions into a set of fundamental vibration modes, each associated with a specific natural frequency. This approach is particularly well-suited for modeling the motion of systems composed of superpositions of harmonic oscillators, such as tree motion [10, 14]. Given an external force $\mathbf f ( t )$ , we model all vertices of the tree mesh as an interconnected mass-spring-damper system $P$ to simulate the response $\mathcal { D } ( t ) = \{ d _ { i } ( t ) \in \mathbb { R } ^ { 3 } \ | \ i \in P \}$ . With this, we can construct the following equation of motion [37]:

$$
M \ddot { d } ( t ) + C \dot { \bf d } ( t ) + K { \bf d } ( t ) = { \bf f } ( t ) ,\tag{5}
$$

where $M , C ,$ and $K$ are the mass, damping, and stiffness matrices, respectively.

To solve this equation, we project it into modal space, resulting in $| P |$ independent equations [8, 23]:

$$
m _ { i } \ddot { q } _ { i } ( t ) + c _ { i } \dot { q } _ { i } ( t ) + k _ { i } q _ { i } ( t ) = f _ { i } ( t ) ,\tag{6}
$$

where $m _ { i } , c _ { i }$ , and $k _ { i }$ correspond to the diagonal elements of the respective matrices. This is a standard second-order differential equation, which can be solved using explicit Euler integration. To perform the integration, we need to specify the initial modal displacement $q _ { i } ( 0 )$ and velocity ${ \dot { q } } _ { i } ( 0 )$ These settings, along with the selection of $M , C ,$ , and $K$ are based on the configurations described in [29].

By solving for the modal responses $q ^ { k } ( t )$ at each natural frequency, we can reconstruct the physical-space response using the corresponding mode shapes:

$$
{ \mathcal { D } } ( t ) = \sum _ { k = 1 } ^ { K } \phi _ { k } \cdot q ^ { k } ( t ) .\tag{7}
$$

Thanks to prior work [8, 23, 29] that has shown the spectrums of particle motion trajectories can be treated as modal bases, we can use the mesh motion spectrums computed in Sec. 3 as the modal shapes $\phi$ to solve the above equation and obtain the interactive dynamic response $\mathcal { D } ( t )$

<!-- image-->  
Figure 3. Comparison with 4D generation methods. We visualize the middle frame of the sequence, where our method better preserves 3D structures. Space-time slices are shown, with vertical and horizontal axes representing time and the spatial profile along the brown line.

## 5. Dataset

To facilitate the learning of complex 3D tree dynamics, we construct 4DTree, a large-scale 4D tree dataset containing 8, 786 animated tree meshes. Each instance provides a 100-frame physically plausible animation sequence that captures realistic and fine-grained wind-induced motion patterns across different tree shapes, sizes, and branching structures.

To create 4D tree data, a straightforward approach is to use commercial physics-based simulation software, but this is time-consuming and impractical for large-scale datasets. Instead, we adopt the method from [45], which models trees as hierarchical branching structures and simulates windinduced motion by treating stems as elastic rods coupled through oscillators, implemented via the Sapling Tree Gen add-on in Blender. Still, this approach involves many sensitive parameters, which, if set improperly, can cause unstable oscillations or unrealistic shapes. To ensure quality and consistency of our dataset, we adopt a three-stage pipeline during production: parameter tuning, automatic validation with scripts, and final visual filtering by human reviewers. Through this process, we construct a clean and diverse 4D tree dataset with complex dynamics. For Detailed procedures and data samples, please refer to the supplementary material.

## 6. Experiments

Implementation. We train our model from scratch without pre-trained models, taking 3.5 days on 8 RTX 4090 GPUs with a batch size of 48. During the first 40,000 iterations, we train the model using only the $\mathcal { L } _ { D M }$ loss. Then, we introduce the $\mathcal { L } _ { L S S }$ loss and continue training for an additional 30,000 iterations. For the $\mathcal { L } _ { L S S }$ loss, we use the 5 nearest neighbors of each point, and both Î± and Î» are set to 0.5. We set the resolution of the sparse voxel spectrum to 1283, with an input resolution of $5 1 2 ^ { 3 }$ for the sparse voxel encoder. This results in voxel latent conditions of dimensionality d = 128 at the $1 2 8 ^ { 3 }$ resolution. When binding 3DGS primitives, we assign five Gaussians per face. We use the AdamW optimizer with an initial learning rate of $1 \times 1 0 ^ { - 4 }$ , which is halved every 20,000 iterations. During inference, we employ DDIM [39] with 100 sampling steps.

Evaluation metrics. To evaluate our method in realworld scenarios, we collect a test set of 13 real-world trees. For evaluation metrics, we follow prior works [6, 52] and use CLIP ViT-B/32 [33] to measure both visual realism and temporal coherence. Specifically, we compute CLIP-I distance as the average CLIP distance between each frame and the input view, and CLIP-T distance as the average CLIP distance between consecutive frames. Furthermore, we conduct a user study on the rendered videos, focusing on four key aspects: motion authenticity (MA), motion complexity (MC), 3D structural consistency (3DSC), and visual quality (VQ). Below, we compare the results of motion generation and interaction simulation, respectively.

## 6.1. Comparison of 3D Animation

For 3D animation, we compare our method with the 4D generation approaches 4DGen [52] and SVD 2.0 [51]. As shown in Fig. 3, the results of 4DGen and SVD 2.0 often contain artifacts in fine structures. Moreover, when the tree structures become more complex, these baselines may fail to converge due to degraded motion generation in their underlying VDMs. For quantitative comparison, the results in Table 1 show that our method consistently outperforms these baselines across all metrics. Additional animation results are provided in the supplementary material, and we strongly recommend viewing the video results on our webpage for a clearer visual assessment.

Table 1. Quantitative comparison of our method and other methods. The upper part is a comparison of 3D animation, and the lower part is a comparison of interactive simulation.
<table><tr><td rowspan="2">Methods</td><td colspan="2">CLIP Score</td><td colspan="5">User Study</td><td rowspan="2">Simulation time (ms/frame)</td></tr><tr><td>CLIP-Iâ</td><td>CLIP-Tâ</td><td>MAâ</td><td>MCâ</td><td>SCâ</td><td>VQâ</td><td>Overallâ</td></tr><tr><td>4DGen [52]</td><td>0.0103</td><td>0.0094</td><td>2.1%</td><td>2.9%</td><td>0.8%</td><td>1.1%</td><td>1.7%</td><td></td></tr><tr><td>SV4D 2.0 [51]</td><td>0.0081</td><td>0.0057</td><td>4.2%</td><td>6.3%</td><td>2.5%</td><td>1.6%</td><td>3.7%</td><td></td></tr><tr><td>Ours</td><td>0.0052</td><td>0.0021</td><td>93.7%</td><td>90.8%</td><td>96.7%</td><td>97.3%</td><td>94.6%</td><td>I</td></tr><tr><td>PhysGaussian [48]</td><td>0.0061</td><td>0.0087</td><td>14.9%</td><td>17.0%</td><td>36.2%</td><td>12.8%</td><td>20.2%</td><td>1,800</td></tr><tr><td>PhysFlow [26]</td><td>0.0047</td><td>0.0025</td><td>34.0%</td><td>38.3%</td><td>34.0%</td><td>21.3%</td><td>31.9%</td><td>15,600</td></tr><tr><td>Ours</td><td>0.0038</td><td>0.0017</td><td>51.1%</td><td>44.7%</td><td>29.8%</td><td>65.9%</td><td>47.9%</td><td>18.22</td></tr></table>

<!-- image-->  
Figure 4. Interactive simulation comparison of different methods. We apply a dragging external force and then visualize the response of the scene, where our approach produces more natural oscillatory motion with finer-grained details. t and T denote the middle and final frames, respectively.

## 6.2. Comparison of Interactive Simulation

For interactive dynamic simulation, we compare against state-of-the-art baselines PhysGaussian [48] and Phys-Flow [26]. As shown in Fig. 4, PhysGaussian and PhysFlow often produce unrealistic plastic deformations. Specifically, PhysGaussian deforms slowly with little rebound, while PhysFlow exhibits partial recovery but still lacks finegrained elasticity at the branch and leaf level, producing overly global responses. In contrast, our method produces natural and elastic motions, with branches and leaves exhibiting distinct behaviors. Quantitative results from four viewpoints and simulation time are reported in Table 1. Our method not only outperforms the baselines on most metrics but also significantly reduces the simulation time. For each frame, our method takes only about 18 ms for simulation, with 13 ms for mesh motion computation via modal analyt slicessis, 2.57 ms for Gaussian deformations calculation, and 2.65 ms for rendering, achieving a real-time interaction. Note that PhysFlow requires additional parameter optimization, which results in significantly longer runtime. More simulation results can be found in the supplementary material.

## 6.3. Comparison with prior physically-based tree animation work

We further conduct a comparative experiment against traditional tree animation methods. As the implementations of several existing methods are not publicly available, we choose [45] as a representative baseline. Note that a major difference from most prior work is that our method can be directly applied to real scanned 3D trees without heavy manual preprocessing, as our voxel-based representation is more robust to scanning errors. For fairness, however, we use the clean 3D tree models used in their algorithm rather than reconstructed trees with scanning artifacts. We perform a user study following the evaluation protocol in Sec. 6, and the results are shown in Table 2. The results show that even when using clean 3D inputs for comparison, our method consistently delivers superior visual quality across multiple evaluation metrics, demonstrating its effectiveness and robustness.

Table 2. Quantitative comparison with traditional tree animation methods.
<table><tr><td>Method</td><td>MAâ</td><td>MCâ</td><td>SCâ</td><td>VQâ</td><td>Overallâ</td></tr><tr><td>Weber [45]</td><td>48.57%</td><td>37.14%</td><td>45.71%</td><td>34.29%</td><td>42.14%</td></tr><tr><td>Ours</td><td>51.43%</td><td>62.86%</td><td>54.29%</td><td>65.71%</td><td>57.86%</td></tr></table>

## 6.4. Ablation Study

The effect of training strategies. We ablate the training strategy in Fig. 5. As shown, directly using $\mathcal { L } _ { D }$ would often cause noticeable artifacts such as geometry scattering and divergence, while joint training with $\mathcal { L } _ { \mathrm { L S S } }$ alleviates but does not eliminate them. In contrast, our two-stage strategy first trains with $\mathcal { L } _ { D }$ for several iterations before introducing $\mathcal { L } _ { \mathrm { L S S } }$ , which effectively resolves these issues and greatly improves generalization.

<!-- image-->  
Figure 5. Ablation of training strategies. Columns 2â4 show the middle frame of sequences generated by each strategy.

The effect of different resolutions. We compare the results of 3D animations at different sparse voxel spectrum resolutions in Table 3. These experiments are conducted on eight GeForce RTX 4090 GPUs, and due to memory constraints, we reduce the batch size as the resolution increases. As shown, the CLIP-I distance first decreases and then increases with increasing resolution. When the resolution exceeds 128, the improvement in CLIP-I becomes marginal, while the training cost continues to rise significantly. Therefore, we select 128 as our final resolution.

Table 3. Ablation of different resolutions
<table><tr><td>Resolution</td><td>Batch Size</td><td>Training Time</td><td>CLIP-Iâ</td></tr><tr><td> $3 2 ^ { 3 }$ </td><td>192</td><td>27h</td><td>0.0097</td></tr><tr><td>643</td><td>96</td><td>43h</td><td>0.0069</td></tr><tr><td>1283</td><td>48</td><td>85h</td><td>0.0039</td></tr><tr><td>2563</td><td>24</td><td>156h</td><td>0.0037</td></tr><tr><td>5123</td><td>12</td><td>261h</td><td>0.0056</td></tr></table>

Moreover, we further analyze the synthetic-to-real gap through the performance degradation observed at a resolution of $5 1 2 ^ { 3 }$ . We find that at such a high resolution, the voxel grid becomes very fine and closely resembles point clouds, introducing a domain gap between training and inference. This is because real mesh vertices are generally noisier than synthetic ones. In contrast, using a resolution of 1283 partially mitigates this issue, as multiple noisy points within the same voxel share the same motion pattern, leading to spatial smoothing that helps bridge the domain gap.

## 7. Limitation and Conclusion

## 7.1. Limitations

Although our method generates realistic 3D motion for real trees, several limitations remain. First, modal analysis is inherently a global linear approximation that shares vibration patterns across the entire object, potentially leading to synchronized motion between spatially distant regions. Second, mesh-driven 3DGS deformation may sometimes introduce artifacts in large deformation areas, which can be alleviated by increasing the number of Gaussians bound to faces in those regions. Finally, our experiments mainly target common immersive motions such as swaying, so the current dataset contains few large-scale deformations. In future work, we plan to augment the dataset with more largedeformation motions to address this limitation.

## 7.2. Conclusion

In this paper, we present DynamicTree, a novel framework for animating 3DGS trees. By introducing the sparse voxel spectrum representation, our method enables efficient longterm real-scanned tree motion generation and real-time dynamic response to external forces. Furthermore, we also introduce a large-scale synthetic 4D tree dataset to support learning-based tree motion generation. Experimental results demonstrate that our approach achieves high-quality tree motion with strong temporal coherence and physical plausibility.

## References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 3

[2] Yasuhiro Akagi. A study on the animations of swaying and breaking trees based on a particle-based simulation. 2012. 2

[3] Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B Lindell. 4d-fy: Text-to-4d generation using hybrid score distillation sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7996â8006, 2024. 2

[4] Hugo Bertiche, Niloy J Mitra, Kuldeep Kulkarni, Chun-Hao P Huang, Tuanfeng Y Wang, Meysam Madadi, Sergio Escalera, and Duygu Ceylan. Blowing in the wind: Cyclenet for human cinemagraphs from still images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 459â468, 2023. 3

[5] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22563â22575, 2023. 3

[6] Ce Chen, Shaoli Huang, Xuelin Chen, Guangyi Chen, Xiaoguang Han, Kun Zhang, and Mingming Gong. Ct4d: Consistent text-to-4d generation with animatable meshes. arXiv preprint arXiv:2408.08342, 2024. 6

[7] Yung-Yu Chuang, Dan B Goldman, Ke Colin Zheng, Brian Curless, David H Salesin, and Richard Szeliski. Animating pictures with stochastic motion textures. In ACM SIG-GRAPH 2005 Papers, pages 853â860. 2005. 3

[8] Abe Davis, Justin G Chen, and FrÃ©do Durand. Image-space modal bases for plausible manipulation of objects in video. ACM Transactions on Graphics (TOG), 34(6):1â7, 2015. 3, 4, 5

[9] Myers Abraham Davis. Visual vibration analysis. PhD thesis, Massachusetts Institute of Technology, 2016. 3

[10] Julien Diener, Mathieu Rodriguez, Lionel Baboud, and Lionel Reveret. Wind projection basis for real-time animation of trees. In Computer graphics forum, pages 533â540. Wiley Online Library, 2009. 3, 5

[11] Linus Franke, Laura Fink, and Marc Stamminger. Vrsplatting: Foveated radiance field rendering via 3d gaussian splatting and neural points. Proc. ACM Comput. Graph. Interact. Tech., 8(1), 2025. 1

[12] Xiangjun Gao, Xiaoyu Li, Yiyu Zhuang, Qi Zhang, Wenbo Hu, Chaopeng Zhang, Yao Yao, Ying Shan, and Long Quan. Mani-gs: Gaussian splatting manipulation with triangular mesh. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21392â21402, 2025. 2

[13] Antoine GuÃ©don and Vincent Lepetit. Sugar: Surfacealigned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering. In Proceedings of

the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5354â5363, 2024. 4

[14] Ralf Habel, Alexander Kusternig, and Michael Wimmer. Physically guided animation of trees. In Computer Graphics Forum, pages 523â532. Wiley Online Library, 2009. 5

[15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840â6851, 2020. 4

[16] Tianyu Huang, Haoze Zhang, Yihan Zeng, Zhilu Zhang, Hui Li, Wangmeng Zuo, and Rynson WH Lau. Dreamphysics: Learning physics-based 3d dynamics with video diffusion priors. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3733â3741, 2025. 3

[17] Junhwa Hur and Stefan Roth. Self-supervised monocular scene flow estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7396â7405, 2020. 14

[18] Chenfanfu Jiang, Craig Schroeder, Andrew Selle, Joseph Teran, and Alexey Stomakhin. The affine particle-in-cell method. ACM Transactions on Graphics (TOG), 34(4):1â10, 2015. 2

[19] Ying Jiang, Chang Yu, Tianyi Xie, Xuan Li, Yutao Feng, Huamin Wang, Minchen Li, Henry Lau, Feng Gao, Yin Yang, et al. Vr-gs: A physical dynamics-aware interactive gaussian splatting system in virtual reality. In ACM SIG-GRAPH 2024 Conference Papers, pages 1â1, 2024. 1

[20] Bernhard Kerbl, Georgios Kopanas, Thomas LeimkÃ¼hler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023. 1

[21] Jiahui Lei, Yijia Weng, Adam W Harley, Leonidas Guibas, and Kostas Daniilidis. Mosca: Dynamic gaussian fusion from casual videos via 4d motion scaffolds. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 6165â6177, 2025. 3

[22] Zhengqi Li, Qianqian Wang, Noah Snavely, and Angjoo Kanazawa. Infinitenature-zero: Learning perpetual view generation of natural scenes from single images. In European Conference on Computer Vision, pages 515â534. Springer, 2022. 4

[23] Zhengqi Li, Richard Tucker, Noah Snavely, and Aleksander Holynski. Generative image dynamics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24142â24153, 2024. 2, 3, 4, 5

[24] Huan Ling, Seung Wook Kim, Antonio Torralba, Sanja Fidler, and Karsten Kreis. Align your gaussians: Text-to-4d with dynamic 3d gaussians and composed diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8576â8588, 2024. 2

[25] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, and Hao Su. One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. Advances in Neural Information Processing Systems, 36:22226â22246, 2023. 3

[26] Zhuoman Liu, Weicai Ye, Yan Luximon, Pengfei Wan, and Di Zhang. Unleashing the potential of multi-modal foundation models and video diffusion for 4d dynamic physical

scene simulation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 11016â11025, 2025. 3, 7

[27] Qiaowei Miao, Jinsheng Quan, Kehan Li, and Yawei Luo. Pla4d: Pixel-level alignments for text-to-4d gaussian splatting. arXiv preprint arXiv:2405.19957, 2024. 2

[28] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 1

[29] Automne Petitjean, Yohan Poirier-Ginter, Ayush Tewari, Guillaume Cordonnier, and George Drettakis. Modalnerf: Neural modal analysis and synthesis for free-viewpoint navigation in dynamically vibrating scenes. In Computer Graphics Forum, page e14888. Wiley Online Library, 2023. 3, 5

[30] SÃ¶ren Pirk, Till Niese, Torsten HÃ¤drich, Bedrich Benes, and Oliver Deussen. Windy trees: Computing stress response for developmental tree models. ACM Transactions on Graphics (TOG), 33(6):1â11, 2014. 1, 2

[31] SÃ¶ren Pirk, MichaÅ Jarz Ëabek, Torsten HÃ¤drich, Dominik L Michels, and Wojciech Palubicki. Interactive wood combustion for botanical tree models. ACM Transactions on Graphics (TOG), 36(6):1â12, 2017.

[32] Ed Quigley, Yue Yu, Jingwei Huang, Winnie Lin, and Ronald Fedkiw. Real-time interactive tree animation. IEEE transactions on visualization and computer graphics, 24(5):1717â 1727, 2017. 1, 2

[33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748â8763. PmLR, 2021. 6

[34] Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao, Gang Zeng, and Ziwei Liu. Dreamgaussian4d: Generative 4d gaussian splatting. arXiv preprint arXiv:2312.17142, 2023. 3

[35] Xuanchi Ren, Jiahui Huang, Xiaohui Zeng, Ken Museth, Sanja Fidler, and Francis Williams. Xcube: Large-scale 3d generative modeling using sparse voxel hierarchies. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4209â4219, 2024. 4

[36] Hannah Schieber, Jacob Young, Tobias Langlotz, Stefanie Zollmann, and Daniel Roth. Semantics-controlled gaussian splatting for outdoor scene reconstruction and rendering in virtual reality. In 2025 IEEE Conference Virtual Reality and 3D User Interfaces (VR), pages 318â328. IEEE, 2025. 1

[37] Ahmed A Shabana. Theory of vibration. Springer, 1991. 5

[38] Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, et al. Text-to-4d dynamic scene generation. In Proceedings of the 40th International Conference on Machine Learning, pages 31915â 31929, 2023. 2, 3

[39] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 6

[40] Olga Sorkine and Marc Alexa. As-rigid-as-possible surface modeling. In Symposium on Geometry processing, pages 109â116. Citeseer, 2007. 5

[41] Qi Sun, Zhiyang Guo, Ziyu Wan, Jing Nathan Yan, Shengming Yin, Wengang Zhou, Jing Liao, and Houqiang Li. Eg4d: Explicit generation of 4d object without score distillation. In The Thirteenth International Conference on Learning Representations. 3

[42] Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. In European Conference on Computer Vision, pages 439â457. Springer, 2024. 3

[43] Joanna Waczynska, Piotr Borycki, SÅawomir Tadeja, Jacek Â´ Tabor, and PrzemysÅaw Spurek. Games: Mesh-based adapting and modification of gaussian splatting. arXiv preprint arXiv:2402.01459, 2024. 2, 5

[44] Qianqian Wang, Vickie Ye, Hang Gao, Weijia Zeng, Jake Austin, Zhengqi Li, and Angjoo Kanazawa. Shape of motion: 4d reconstruction from a single video. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9660â9672, 2025. 3

[45] Jason Weber and Joseph Penn. Creation and rendering of realistic trees. In Proceedings of the 22nd annual conference on Computer graphics and interactive techniques, pages 119â128, 1995. 2, 6, 7, 8, 12

[46] Francis Williams, Jiahui Huang, Jonathan Swartz, Gergely Klar, Vijay Thakkar, Matthew Cong, Xuanchi Ren, Ruilong Li, Clement Fuji-Tsang, Sanja Fidler, Eftychios Sifakis, and Ken Museth. fvdb: A deep-learning framework for sparse, large-scale, and high-performance spatial intelligence. ACM Transactions on Graphics (TOG), 43(4):133:1â 133:15, 2024. 4

[47] Rundi Wu, Ruiqi Gao, Ben Poole, Alex Trevithick, Changxi Zheng, Jonathan T Barron, and Aleksander Holynski. Cat4d: Create anything in 4d with multi-view video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26057â26068, 2025. 3

[48] Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu Jiang. Physgaussian: Physicsintegrated 3d gaussians for generative dynamics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4389â4398, 2024. 2, 3, 7

[49] Yiming Xie, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, and Varun Jampani. Sv4d: Dynamic 3d content generation with multi-frame and multi-view consistency. In The Thirteenth International Conference on Learning Representations. 3

[50] Dejia Xu, Hanwen Liang, Neel P Bhatt, Hezhen Hu, Hanxue Liang, Konstantinos N Plataniotis, and Zhangyang Wang. Comp4d: Llm-guided compositional 4d scene generation. arXiv preprint arXiv:2403.16993, 2024. 2

[51] Chun-Han Yao, Yiming Xie, Vikram Voleti, Huaizu Jiang, and Varun Jampani. Sv4d 2.0: Enhancing spatio-temporal consistency in multi-view video diffusion for high-quality 4d generation. arXiv preprint arXiv:2503.16396, 2025. 3, 6, 7

[52] Yuyang Yin, Dejia Xu, Zhangyang Wang, Yao Zhao, and Yunchao Wei. 4dgen: Grounded 4d content generation with spatial-temporal consistency. arXiv preprint arXiv:2312.17225, 2023. 2, 3, 6, 7

[53] Yu-Jie Yuan, Leif Kobbelt, Jiwen Liu, Yuan Zhang, Pengfei Wan, Yu-Kun Lai, and Lin Gao. 4dynamic: Text-to-4d generation with hybrid priors. arXiv preprint arXiv:2407.12684, 2024. 3

[54] Tianyuan Zhang, Hong-Xing Yu, Rundi Wu, Brandon Y Feng, Changxi Zheng, Noah Snavely, Jiajun Wu, and William T Freeman. Physdreamer: Physics-based interaction with 3d objects via video generation. In European Conference on Computer Vision, pages 388â406. Springer, 2024. 3

[55] Yili Zhao and Jernej Barbic. Interactive authoring of Ë simulation-ready plants. ACM Transactions on Graphics (TOG), 32(4):1â12, 2013. 2

[56] Yufeng Zheng, Xueting Li, Koki Nagano, Sifei Liu, Otmar Hilliges, and Shalini De Mello. A unified approach for textand image-guided 4d scene generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7300â7309, 2024. 2, 3

## A. Dataset

As discussed in the paper, we implement the method of [45] as a plugin in Blender. However, generating a large-scale 4D tree dataset using this method remains challenging, as it involves numerous parameters and random sampling often produces invalid or unrealistic 4D trees. To ensure data quality and consistency, we adopt a three-stage pipeline to construct our dataset:

1. Parameter Tuning: Trees are controlled by many shape parameters. Fully random sampling over all of them tends to generate irregular or unrealistic trees, which can harm network training. Instead, we manually select key parameters such as branch count, height, branching angle, leaf count, etc., for stochastic variation. Other parameters are kept within small perturbation ranges. This approach ensures diversity while avoiding extreme or implausible deformations.

2. Automatic Filtering: After generating approximately 10,000 trees using the above strategy, we observe that some samples exhibit undesirable high-frequency oscillations, such as rapid back-and-forth motion at the root or excessive shaking in small branches. To filter out these cases, we apply the Fast Fourier Transform to each motion sequence and remove samples where the highfrequency components exceed a threshold.

3. Manual Curation: Finally, we perform visual inspection to eliminate edge cases such as unnatural branch clustering or physically implausible motion patterns.

Through this process, we curate a final set of 8,786 4D trees, with selected examples visualized in Fig. 6. For each tree, we first apply the FFT to its motion and then voxelize it. The spectrum of vertices within the same voxel is averaged to produce the final sparse voxel spectrum representation.

## B. Network

The sparse encoder and sparse voxel diffusion U-Net are adapted from the basic modules proposed in XCube to better fit our conditioning input and spectral output. We report key parameter settings of these two components in Table 4.

Table 4. Architecture Parameters
<table><tr><td>Parameter</td><td>Sparse Encoder</td><td>Voxel Diffusion</td></tr><tr><td>Base channels</td><td>32</td><td>128</td></tr><tr><td>Depth</td><td>3</td><td>2</td></tr><tr><td>Channels multiple</td><td>-</td><td>[1, 2, 4, 4]</td></tr><tr><td>Head</td><td></td><td>8</td></tr><tr><td>Attention Resolution</td><td>-</td><td>[4,8]</td></tr></table>

## C. Visualization results

Further visualizations and analytical details of the realworld 3D tree animations and interactive dynamic simulations are presented in Fig. 8 and Fig. 7. Moreover, to facilitate a comprehensive perceptual and qualitative evaluation of our method, we strongly recommend reviewing the video results on our webpage.

<!-- image-->  
Figure 6. Examples from our 4DTree dataset. For clarity of visualization, the leaves and trunk are rendered using two simplified material configurations.

<!-- image-->  
Figure 7. More results of interactive dynamic simulation. Our method can support interactive simulations involving forces with varying magnitudes and directions.

<!-- image-->  
Figure 8. More results of real-tree 3D animation. For each scene, we visualize the space-time slices of depth and RGB videos from two viewpoints. We also show the scene flow of three mesh point cloud frames $( t _ { 1 } = 3 0 , t _ { 2 } = 5 0 , t _ { 3 } = 8 0 )$ in the generated sequence, with color coding following the strategy used in [17], where the movements (u, v) along the x and z directions are encoded using standard optical flow coloring.