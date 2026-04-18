# Active View Selection with Perturbed Gaussian Ensemble for Tomographic Reconstruction

Yulun Wu1 , Ruyi Zha2 , Wei Cao1 ,

Yingying Li1 , Yuanhao Cai3,â , and Yaoyao Liu1

{yulun5, weicao3, yl101, lyy}@illinois.edu ruyi.zha@anu.edu.au ycai51@jh.edu

1 University of Illinois Urbana-Champaign

2 Australian National University

3 Johns Hopkins University

Abstract. Sparse-view computed tomography (CT) is critical for reducing radiation exposure to patients. Recent advances in radiative 3D Gaussian Splatting (3DGS) have enabled fast and accurate sparse-view CT reconstruction. Despite these algorithmic advancements, practical reconstruction fidelity remains fundamentally bounded by the quality of the captured data, raising the crucial yet underexplored problem of X-ray active view selection. Existing active view selection methods are primarily designed for natural-light scenes and fail to capture the unique geometric ambiguities and physical attenuation properties inherent in X-ray imaging. In this paper, we present Perturbed Gaussian Ensemble, an active view selection framework that integrates uncertainty modeling with sequential decision-making, tailored for X-ray Gaussian Splatting. Specifically, we identify low-density Gaussian primitives that are likely to be uncertain and apply stochastic density scaling to construct an ensemble of plausible Gaussian density fields. For each candidate projection, we measure the structural variance of the ensemble predictions and select the one with the highest variance as the next best view. Extensive experimental results on arbitrary-trajectory CT benchmarks demonstrate that our density-guided perturbation strategy effectively eliminates geometric artifacts and consistently outperforms existing baselines in progressive tomographic reconstruction under unified view selection protocols.1

Keywords: CT Reconstruction Â· Gaussian Splatting Â· Active Learning

## 1 Introduction

X-ray computed tomography (CT) is an indispensable non-invasive imaging modality widely utilized in medical diagnosis, industrial inspection, and scientific research [73, 77]. During a CT scan, an X-ray machine captures multi-angle 2D projections that measure ray attenuation through the material. Tomographic reconstruction can then be performed to recover high-fidelity 3D anatomical structures from the 2D projections. However, the ionizing radiation associated with prolonged X-ray exposure poses significant health risks.

<!-- image-->  
PSNR: 29.45SSIM: 0.847

<!-- image-->  
PSNR: 33.17SSIM: 0.875

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
(a)

<!-- image-->

<!-- image-->  
(b)  
Fig. 1: (a) We compare our approach against the state-of-the-art 3DGS-based active view selection method, FisherRF [24], evaluating both quantitative metrics (3D PSNRâ and SSIMâ) and qualitative visual fidelity, including zoomed-in details. Our method achieves the highest reconstruction quality and best preserves fine structural details. (b) We plot the training iterations vs. the average 3D PSNRâ (dB) and SSIMâ on the synthetic dataset. As demonstrated, our proposed approach consistently delivers superior reconstruction fidelity throughout the progressive reconstruction process.

To mitigate these risks, sparse-view CT has emerged as a critical scanning paradigm, aiming to maximize the reconstruction quality with a limited number of projection angles. Mathematically, this reduction in data acquisition transforms the tomographic reconstruction into a highly ill-posed inverse problem. Existing algorithms and solvers often struggle under such sparse conditions.

Recent breakthroughs in neural rendering, particularly 3D Gaussian Splatting (3DGS) [30], have demonstrated a remarkable performance superiority in both sparse-view novel view synthesis [7,34,76,78] and surface reconstruction [22, 23, 64]. Building on this, several methods [5, 71, 73] have successfully adapted 3DGS to X-ray imaging, enabling rapid and highly accurate volumetric reconstructions. However, despite these algorithmic leaps, practical reconstruction quality remains fundamentally bottlenecked by the captured data. Consequently, how to accurately select scanning viewpoints under a limited view-budget, so as to ensure both global silhouette coverage and the capture of local structural details, has emerged as a critical challenge in sparse-view tomography.

While crucial, active view selection for X-ray Gaussian Splatting remains largely unexplored. Existing view selection strategies [24, 50] are predominantly designed for natural-light scenes, relying heavily on surface occlusions and viewdependent specularities to estimate uncertainty or information gain. In stark contrast, X-ray imaging operates on the Beer-Lambert law [28], where the projection is a purely linear integral of the continuous density field along the ray path, without any occlusion. Furthermore, X-ray attenuation is inherently isotropic, meaning the constituent Gaussian primitives have no spherical harmonic (SH) parameters. Consequently, methods that depend on view-dependent color gradients and assume sparse, occlusion-bounded ray interactions fail to accurately capture the volumetric ambiguity in CT. As a result, these methods struggle to distinguish between the stretched artifacts and true high-density structures, often yielding redundant view selections that fail to provide the orthogonal constraints necessary to eliminate such artifacts.

In this study, we present Perturbed Gaussian Ensemble, a novel active view selection framework for progressive reconstruction tailored specifically to X-ray Gaussian Splatting. Our core intuition is that under sparse-view constraints, geometric ambiguities typically manifest as fragile structures, such as uncertain boundaries and needle-like artifacts, whose projections vary dramatically when observed from informative, previously unseen angles. A valid next best view is the one that maximizes the exposure of this underlying structural instability.

To operationalize this insight, we utilize low-density Gaussian primitives as a proxy for uncertainty-prone regions, as they typically correspond to underconstrained boundaries, background noise, or degenerated artifact tails. By applying stochastic perturbations to the densities of these specific primitives, we construct an ensemble of plausible Gaussian density fields. For each candidate viewpoint, we render predictions across this ensemble and quantify their structural disagreement using the variance of the Structural Similarity Index Measure (SSIM) [62]. A high structural variance indicates that minor perturbations in uncertain regions provoke substantial structural discrepancies in that specific projection, marking the viewpoint as highly informative for resolving existing ambiguities. We therefore select the viewpoint that maximizes this structural variance as the optimal next best view.

We comprehensively evaluate our method against 2D-based paradigms, 3Dbased paradigms, and conventional rule-based heuristics under two different view selection protocols within a hemispherical scanning search space. Experimental results demonstrate that our density-guided perturbation strategy effectively eliminates geometric artifacts and consistently outperforms existing baselines in progressive tomographic reconstruction, as illustrated in Fig. 1.

Our contribution can be summarized as follows:

We propose a novel active view selection and progressive reconstruction framework designed for X-ray Gaussian Splatting. Our work bridges the gap between active learning and explicit radiative fields, addressing the unique physical and geometric challenges of sparse-view computed tomography.

â We introduce a novel uncertainty quantification strategy based on the Perturbed Gaussian Ensemble. By applying stochastic density perturbations to low-density primitives that are highly susceptible to geometric degradation and measuring the structural disagreement in projection space, our method accurately localizes epistemic uncertainty and predicts the next best view.

â We establish an active view selection and progressive reconstruction benchmark for radiative Gaussian Splatting by adapting and evaluating previous state-of-the-art baselines. Extensive experiments demonstrate that our approach consistently outperforms existing paradigms, achieving superior volumetric reconstruction and novel view synthesis quality.

## 2 Related Work

Tomographic Reconstruction aims to recover the 3D internal density field of an object from 2D X-ray projections. Traditional analytical methods [16, 70] suffer from severe streak artifacts and noise amplification under sparse-view conditions. Iterative reconstruction algorithms [2, 48, 56, 59] rely on optimization over iterations; however, they are computationally expensive and tend to oversmooth fine structural details. While supervised deep learning approaches [1, 3, 12, 20, 25, 33, 37â39, 46, 69] leverage semantic priors to enhance image quality, their generalization capabilities often remain constrained. NeRF-based frameworks [6,54,58,72,74] model the continuous density field using coordinate-based MLPs optimized purely by photometric losses from sparse projections. Despite achieving high-fidelity reconstructions, their training and rendering processes are prohibitively slow. The emergence of 3D Gaussian Splatting (3DGS) [30] offers a compelling alternative due to its explicit parameterization and highly parallelized rasterization. Early adaptations to X-ray imaging [5,19] focus primarily on novel view synthesis rather than direct volume retrieval. R2-Gaussian [73] achieves a critical breakthrough by identifying and rectifying the omission of a covariancerelated scaling factor during 3D-to-2D projection. By introducing tailored radiative Gaussian kernels and a differentiable voxelizer, it enables direct, bias-free, and rapid static tomographic reconstruction. Subsequent works [35, 47, 71, 75] further explore the sparse-view CT reconstruction paradigm or extend the radiative splatting framework into the temporal domain. However, active view selection for CT remains largely unexplored. Therefore, this study undertakes a more in-depth investigation and introduces a targeted solution to this problem.

Active View Selection (AVS) or next best view (NBV) planning aims to incrementally [15, 18, 40â45] determine the most informative viewpoints for scene reconstruction to minimize data acquisition costs. It originates from robotics research [13, 57] and sees extensive exploration in 3D reconstruction [10, 11, 14, 17, 26, 52]. With the rise of neural rendering, recent works adapt AVS to neural radiance fields (NeRF) [49]. NeRF-based methods [50, 65, 67] quantify uncertainty via variance estimation to guide view acquisition, but they remain computationally intensive and primarily apply to synthetic settings. The advent of 3D Gaussian Splatting (3DGS) [30] significantly accelerates active view selection and reconstruction due to its explicit representation and highly efficient rasterization. 3DGS-based methods [24,27,36,63,66] adopt uncertainty or visibility-driven selection and show great promise for efficiency. However, current gradient-based algorithms assume surface-based occlusion and heavily rely on view-dependent color parameters. In contrast, X-ray imaging operates on a transmission model governed by the Beer-Lambert law [28], where projections serve as linear integrals of the density field without occlusion. This transmissive nature causes high spatial coupling among Gaussians along the ray, severely violating the diagonal approximation assumptions of gradient-based methods. To address this gap, we abandon the gradient-based heuristic and introduce a forward parameter perturbation strategy specifically tailored to the explicit density parameters of radiative Gaussians.

<!-- image-->

<!-- image-->  
(b) Ensemble-Based

<!-- image-->  
(c) Perturbed Gaussian Ensemble (Ours)  
Fig. 2: Comparison of active view selection paradigms for X-ray Gaussian Splatting. (a) Gradient-based method [24] estimates the expected information gain (EIG) of candidate views by computing a Fisher Information Matrix (FIM) via backpropagation, but suffers from gradient coupling and the absence of view-dependent parameters. (b) Ensemble-based method quantifies epistemic uncertainty by rendering disagreement across multiple Gaussian representations initialized with different random seeds, resulting in prohibitively high computational burden. (c) Our Perturbed Gaussian Ensemble introduces density-guided parameter perturbations to efficiently construct an ensemble, wherein uncertain primitives exhibit pronounced behavioral randomness. By calculating the structural variance of the rendered projections, our approach achieves superior uncertainty modeling and enables more effective active view selection.

## 3 Method

Given a sparse initial collection of measured X-ray projections ${ \mathcal { T } } _ { \mathrm { i n i t } } = \{ \mathbf { I } _ { \mathrm { i n i t } } ^ { ( i ) } \in$ $\mathbb { R } ^ { H \times W } \} _ { i = 1 , \dots , N _ { \mathrm { i n i t } } }$ , the aim is to iteratively select the next best view (NBV) from a pool of candidate scanner poses and incorporate the corresponding measurement into the training-view set until a predefined target number of observations $N _ { \mathrm { t a r g e t } }$ is reached. Throughout this process, we progressively reconstruct the 3D volume to maximize the final reconstruction quality. To achieve this, we exploit the physical properties of radiative Gaussian Splatting to introduce a novel active view selection framework. As illustrated in Fig. 2, our approach models viewpoint uncertainty through the use of Perturbed Gaussian Ensemble.

In this section, we present our proposed active view selection framework. First, we introduce the radiative Gaussian Splatting model and establish the theoretical basis in Sec. 3.1. Next, we detail the construction of Perturbed Gaussian Ensemble through a density-guided parameter perturbation strategy in Sec. 3.2. Finally, Sec. 3.3 formulates our view selection mechanism, which quantifies epistemic uncertainty by evaluating structural variance in the projection space.

## 3.1 Preliminary: Radiative Gaussian Splatting

To facilitate efficient and physically correct tomographic reconstruction, our active view selection mechanism is built on a radiative Gaussian Splatting framework, which reformulates 3D Gaussian Splatting (3DGS) [30] for transmission imaging. Following [73], we model the objectâs 3D density field Ï as a mixture of M radiative Gaussian kernels. Each kernel $G _ { i }$ acts as a local volumetric density primitive parameterized by a central density scalar $\rho _ { i } .$ a mean position $\mathbf { p } _ { i } \in \mathbb { R } ^ { 3 }$ and a 3D covariance matrix $\pmb { \Sigma } _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ . The density field at a spatial location $\mathbf { x } \in \mathbb { R } ^ { 3 }$ is formulated as

$$
\sigma ( { \mathbf x } ) = \sum _ { i = 1 } ^ { M } G _ { i } ^ { 3 } ( { \mathbf x } \mid { \boldsymbol \rho } _ { i } , { \mathbf p } _ { i } , { \boldsymbol \Sigma } _ { i } ) = \sum _ { i = 1 } ^ { M } { \rho } _ { i } \exp \left( - \frac { 1 } { 2 } ( { \mathbf x } - { \mathbf p } _ { i } ) ^ { \top } { \boldsymbol \Sigma } _ { i } ^ { - 1 } ( { \mathbf x } - { \mathbf p } _ { i } ) \right) .\tag{1}
$$

The rendering process simulates the physical acquisition of X-ray projections. Due to the linearity of the integral operator, the total log-space projection is the sum of the line integrals of individual Gaussians:

$$
I _ { \mathrm { r } } ( \mathbf { r } ) = \sum _ { i = 1 } ^ { M } G _ { i } ^ { 2 } ( \hat { \mathbf { x } } \mid \rho _ { i } \sqrt { \frac { 2 \pi \vert \pmb { \Sigma } _ { i } \vert } { \vert \hat { \mathbf { \Sigma } } \vert _ { i } \vert } } , \hat { \mathbf { p } } _ { i } , \hat { \mathbf { \Sigma } } _ { i } ) ,\tag{2}
$$

where $I _ { \mathrm { r } } ( \mathbf { r } )$ is the rendered pixel value; $\hat { \mathbf { x } } \in \mathbb { R } ^ { 2 } , \hat { \mathbf { p } } \in \mathbb { R } ^ { 2 } , \hat { \mathbf { \Sigma } } \in \mathbb { R } ^ { 2 \times 2 }$ are obtained by projecting the Gaussians onto the image plane from a viewpoint. The term $\sqrt { 2 \pi | \boldsymbol { \Sigma } _ { i } | / | \hat { \boldsymbol { \Sigma } } _ { i } | }$ (where |Â·| denotes the matrix determinant) acts as a normalization factor that ensures conservation of the integrated density mass, rendering the optimization of $\rho _ { i }$ physically meaningful and view-consistent.

Radiative Gaussians are optimized by photometric L1 loss $\mathcal { L } _ { 1 }$ , D-SSIM loss $\mathcal { L } _ { \mathrm { s s i m } }$ [62], and 3D total variation regularization ${ \mathcal { L } } _ { \mathrm { t v } }$ [55]. The overall loss function is defined as

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { 1 } ( \mathbf { I } _ { \mathrm { r } } , \mathbf { I } _ { \mathrm { m } } ) + \lambda _ { 1 } \mathcal { L } _ { \mathrm { s s i m } } ( \mathbf { I } _ { \mathrm { r } } , \mathbf { I } _ { \mathrm { m } } ) + \lambda _ { 2 } \mathcal { L } _ { \mathrm { t v } } , } \end{array}\tag{3}
$$

where $\mathbf { I } _ { \mathrm { r } } , \mathbf { I } _ { \mathrm { m } }$ are the rendered and measured projection. $\lambda _ { 1 } , \lambda _ { 2 }$ are weight terms.

## 3.2 Perturbed Gaussian Ensemble

Due to the transmissive and isotropic nature of X-rays, active view selection methods designed for natural-light scenes, such as FisherRF [24], perform poorly in X-ray CT settings. To maintain real-time performance, FisherRF evaluates the expected information gain (EIG) using a diagonal approximation of the Fisher Information Matrix (FIM). While ignoring parameter correlations is valid for natural-light rendering, where pixels are dominated by a few front-most Gaussian primitives, this assumption fails in X-ray imaging. The transmissive nature of X-rays highly couples Gaussians along a ray, causing substantial mathematical bias in the EIG under a diagonal FIM.

To circumvent the computational intractability of evaluating a dense FIM and the inherent inaccuracies of its diagonal approximation, we abandon the gradient-based heuristic. Instead, we propose a forward, sampling-based approach that directly explores the coupled geometric uncertainty.

Uncertainty Quantification via Rendering Disagreement. In sparse-view CT reconstruction, the lack of sufficient multi-view geometry constraints leads to severe ill-posedness, introducing significant uncertainty into the optimization of Gaussian fields. Let D denote the training data and Î¸ represent the parameters of a radiative Gaussian Splatting model. For an arbitrary scanner pose v, the predicted projection $\mathbf { I } ( \mathbf { v } )$ is obtained through the rendering process:

$$
\mathbf { I } ( \mathbf { v } ) = f ( \mathbf { v } ; \theta ) .\tag{4}
$$

During optimization, we minimize the total loss to obtain the optimal model parameters:

$$
\boldsymbol { \hat { \theta } } = \arg \operatorname* { m i n } _ { \boldsymbol { \theta } } \mathcal { L } ( \boldsymbol { \theta } ; \mathcal { D } ) + \lambda R ( \boldsymbol { \theta } ) ,\tag{5}
$$

where $R ( \theta )$ represents the regularization term. Due to the highly non-convex nature of the objective function, the optimization problem admits multiple minima. Consequently, when initialized with different random seeds $\{ k _ { i } \} _ { i = 1 , \dots , N }$ , the model converges to distinct local optima:

$$
{ \hat { \theta } } _ { i } \sim q ( \theta \mid D ) ,\tag{6}
$$

where $q ( \theta \mid D )$ represents an empirical approximation of the true posterior $p ( \theta \mid \mathcal { D } )$ . Thus, the posterior predictive distribution at viewpoint v can be approximated via Monte Carlo integration over the ensemble:

$$
p ( \mathbf { I } ( \mathbf { v } ) \mid \mathcal { D } ) \approx \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \delta \left( \mathbf { I } ( \mathbf { v } ) - \mathbf { I } _ { i } ( \mathbf { v } ) \right) , \quad \mathbf { I } _ { i } ( \mathbf { v } ) = f ( \mathbf { v } ; \hat { \theta } _ { i } ) ,\tag{7}
$$

where $\delta ( \cdot )$ denotes the Dirac delta function. Therefore, the disagreement in the rendered projections across an ensemble of N explicit Gaussian representations $\{ \mathcal { G } _ { i } \} _ { i = } ^ { N }$ initialized with different random seeds $\{ k _ { i } \} _ { i = 1 , \dots , N }$ effectively captures

the epistemic uncertainty at a given view. We quantify this uncertainty using the sample variance:

$$
\mathrm { V a r } \left[ \mathbf { I } ( \mathbf { v } ) \right] = \frac { 1 } { N - 1 } \sum _ { i = 1 } ^ { N } \left( \mathbf { I } _ { i } ( \mathbf { v } ) - \bar { \mathbf { I } } ( \mathbf { v } ) \right) ^ { 2 } , \quad \bar { \mathbf { I } } ( \mathbf { v } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { I } _ { i } ( \mathbf { v } ) .\tag{8}
$$

Density-Guided Parameter Perturbation. Training an ensemble of Gaussian models is an intuitive and effective approach for viewpoint uncertainty modeling. However, this strategy requires training and storing N independent Gaussian Splatting models, leading to a linear increase in computational and memory complexity, which is often prohibitive in practical applications. To address this, we propose Perturbed Gaussian Ensemble, a simulation-based approach that leverages the intrinsic properties of the radiative Gaussian model.

Specifically, the variance in the rendered outputs across an ensemble at a given viewpoint is fundamentally induced by the randomness in the learned parameters of the individual Gaussian primitives. Therefore, a more efficient strategy is to train only a single Gaussian model ${ \mathcal { G } } = \{ G _ { i } \} _ { i = 1 , \dots , M }$ and, during uncertainty evaluation for candidate viewpoints, independently perturb the parameters of the Gaussian representation with repeated noise injections. This efficiently emulates an ensemble that would otherwise be obtained by training with different random seeds:

$$
\mathcal { G } _ { i } = \{ G _ { i , j } \} , \quad G _ { i , j } = G _ { j } \oplus \varDelta _ { i , j } , \quad i = 1 , \ldots , N , \quad j = 1 , \ldots , M .\tag{9}
$$

To ensure that the rendering disagreement of this perturbed ensemble accurately reflects the epistemic uncertainty arising from the actual training process, the noise injected into each individual Gaussian primitive should be conditioned on its specific uncertainty level.

According to Eqs. (1) and (2), the contribution of a Gaussian primitive during rendering is heavily determined by its density. High-density primitives typically correspond to real, well-defined solid structures (e.g., bones and dense organs). Because these structures strongly attenuate X-rays, they are usually well-constrained even under sparse-view settings. Consequently, the model exhibits high confidence in these regions.

In contrast, low-density primitives often lie near object boundaries (e.g., softtissue interfaces) or constitute background noise introduced to fit only a small number of rays, thereby exhibiting higher uncertainty. Moreover, due to insufficient multi-view constraints, the Gaussians tend to overfit the training views during optimization, giving rise to extremely elongated, needle-like artifacts or numerous tiny, low-density Gaussians distributed along the ray paths. These spurious components likewise exhibit low density and high uncertainty.

Motivated by this observation, we isolate a vulnerable Gaussian subset $\mathcal { G } _ { \mathrm { l o w } }$ comprising a specific fraction Î± (e.g., 10%) of the total Gaussians in $\mathcal { G }$ with the lowest density values. We then stochastically perturb the density parameters of this subset to probe the stability of the reconstructed scene structure. Specifically, we generate an ensemble of N independently perturbed models. For the i-th ensemble member $\mathcal { G } _ { i } ( i = 1 , \ldots , N )$ , the density $\rho _ { i , j }$ of a perturbed Gaussian primitive $G _ { i , j }$ derived from $G _ { j } \in \mathcal { G } \ ( j = 1 , \ldots , M )$ is defined as:

$$
\rho _ { i , j } = \left\{ { \begin{array} { l l } { \rho _ { j } \cdot ( 1 + \epsilon _ { i , j } ) , } & { { \mathrm { i f ~ } } G _ { j } \in { \mathcal { G } } _ { \mathrm { l o w } } } \\ { \rho _ { j } , } & { { \mathrm { o t h e r w i s e } } } \end{array} } , \right.\tag{10}
$$

where $\epsilon _ { i , j }$ is a random scaling factor sampled from a uniform distribution:

$$
\epsilon _ { i , j } \sim \mathrm { U n i f o r m } ( - \beta , \beta ) , \quad \beta > 0 .\tag{11}
$$

This targeted density scaling effectively injects variations into the most ambiguous regions of the volume while preserving the high-density, high-confidence anatomical structures.

## 3.3 View Selection by Structural Variance

The core objective of active view selection is to identify the candidate viewpoint that provides the most informative structural constraints to resolve existing geometric ambiguities. For an arbitrary candidate viewpoint $\mathbf { v } ,$ rendered projections are generated using the Perturbed Gaussian Ensemble $\{ \mathcal { G } , \mathcal { G } _ { 1 } , \ldots , \mathcal { G } _ { N } \}$ via the rendering function:

$$
\mathbf { I } ( \mathbf { v } ) = f \left( \mathbf { v } ; \theta ( \mathcal { G } ) \right) , \quad \mathbf { I } _ { i } ( \mathbf { v } ) = f \left( \mathbf { v } ; \theta ( \mathcal { G } _ { i } ) \right) , \quad i = 1 , \ldots , N .\tag{12}
$$

To quantify the epistemic uncertainty at viewpoint $\mathbf { v } ,$ we leverage the Structural Similarity Index Measure (SSIM) [62] in the projection space. This allows us to evaluate the macroscopic structural disagreement induced by our parameter perturbations:

$$
s _ { i } ( \mathbf v ) = \mathrm { S S I M } \left( \mathbf I ( \mathbf v ) , \mathbf I _ { i } ( \mathbf v ) \right) .\tag{13}
$$

The uncertainty score $u ( \mathbf { v } )$ for the candidate viewpoint is formulated as the sample variance of these N structural similarity scores:

$$
u ( \mathbf { v } ) = \frac { 1 } { N - 1 } \sum _ { i = 1 } ^ { N } \left( s _ { i } ( \mathbf { v } ) - \bar { s } ( \mathbf { v } ) \right) ^ { 2 } ,\tag{14}
$$

where $\bar { s } ( \mathbf { v } )$ denotes the mean of the N SSIM scores.

At each view selection iteration $t ,$ a new perturbed ensemble is generated from the most recently updated Gaussian representation $\mathcal { G } ^ { ( t ) }$ . For every viewpoint from the candidate pool $\mathcal { V } _ { \mathrm { c a n d } } ^ { ( t ) }$ , we calculate the epistemic uncertainty score using this ensemble. Finally, the viewpoint exhibiting the highest uncertainty is selected as the next best view:

$$
\mathbf { v } ^ { * ( t ) } = \arg \operatorname* { m a x } _ { \mathbf { v } \in \mathcal { V } _ { \mathrm { c a n d } } ^ { ( t ) } } u ( \mathbf { v } ) .\tag{15}
$$

The corresponding physical measurement from $\mathbf { v } ^ { * ( t ) }$ is then acquired and $\mathrm { a p - }$ pended to the training set for the subsequent phase of progressive optimization.

<table><tr><td rowspan="2">Category</td><td rowspan="2">Method</td><td colspan="2">24-view</td><td colspan="2">36-view</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>PSNRâ</td><td>SSIMâ</td></tr><tr><td colspan="7">Synthetic Dataset</td></tr><tr><td rowspan="2">Rule-based</td><td>Random</td><td>32.629</td><td>0.881</td><td>34.823</td><td>0.915</td></tr><tr><td>FPS</td><td>33.508</td><td>0.891</td><td>35.367</td><td>0.919</td></tr><tr><td rowspan="2">2D-based</td><td>TOPIQ [9] MANIQA [68]</td><td>33.000</td><td>0.882 0.882</td><td>34.951</td><td>0.912</td></tr><tr><td>MUSIQ [29]</td><td>32.319 33.083</td><td>0.886</td><td>34.543 35.030</td><td>0.913</td></tr><tr><td rowspan="2">3D-based</td><td>FisherRF [24]</td><td>33.347</td><td>0.887</td><td>35.551</td><td>0.916 0.919</td></tr><tr><td>Ours</td><td>34.078</td><td>0.896</td><td>36.226</td><td>0.926</td></tr><tr><td colspan="6">Real-World</td></tr><tr><td rowspan="2">Rule-based</td><td>Random</td><td>36.112</td><td>Dataset 0.903</td><td></td><td></td></tr><tr><td>FPS</td><td>36.134</td><td></td><td>36.765</td><td>0.921</td></tr><tr><td rowspan="3">2D-based</td><td>TOPIQ [9]</td><td>36.208</td><td>0.905</td><td>36.898</td><td>0.925</td></tr><tr><td>MANIQA [68]</td><td></td><td>0.906</td><td>37.167</td><td>0.928</td></tr><tr><td>MUSIQ [29]</td><td>36.222 36.257</td><td>0.906 0.906</td><td>37.187 37.119</td><td>0.929</td></tr><tr><td rowspan="2">3D-based</td><td>FisherRF [24]</td><td>36.205</td><td>0.902</td><td>37.258</td><td>0.925 0.926</td></tr><tr><td>Ours</td><td>36.399</td><td>0.909</td><td>37.480</td><td>0.932</td></tr></table>

Table 1: Quantitative comparisons of 3D tomographic reconstruction across different view selection strategies under two protocols. Performance is evaluated in terms of 3D PSNRâ (dB) and 3D SSIMâ. The best and second-best results are highlighted. Random and FPS denote random view selection and Farthest Point Sampling, respectively.

## 4 Experiments

## 4.1 Experimental Settings

Datasets. Following the experimental settings of R2-Gaussian [73], we evaluate our method on both synthetic and real-world datasets. The synthetic benchmark comprises 15 CT volumes aggregated from various public datasets [4, 32, 53, 61]. For real-world evaluation, we utilize three cases from the FIPS dataset [60], reconstructing pseudo-ground-truth (pseudo-GT) volumes via FDK [16]. We adopt an object-centric hemispherical acquisition trajectory rather than a conventional circular orbit. The candidate viewpoint pool for active selection consists of 448 poses uniformly sampled across two concentric hemispheres at different radii, with all optical axes oriented toward the volume center. For novel view synthesis evaluation, 103 test viewpoints are randomly sampled from a single hemisphere per scene. All GT and pseudo-GT projections are simulated using DiffDRR [21].

Protocols. We initialize the Gaussian primitives with 50k random points and train the radiative GS model for 30k iterations per scene under the default configuration. Following FisherRF [24], active view selection is performed at progressively increasing iteration intervals. To ensure stable optimization, we schedule the final view selection prior to the termination of the GS densification phase. At each selection step, a single optimal view is acquired and appended to the training set. We evaluate our method and all baselines under two final view-budget protocols: $N _ { \mathrm { t a r g e t } } = 2 4$ and 36 (inclusive of two initial views).

<!-- image-->  
Fig. 3: Visual comparisons of the reconstructed 3D volumes using different view selection strategies. The 3D PSNRâ (dB) for each scene is displayed at the top-left corner of the corresponding image. Our approach consistently achieves the highest reconstruction quality and better preserves fine structural details.

Baselines. Previous active view selection strategies generally fall into two paradigms: 2D-based approaches that evaluate rendered image quality, and 3Dbased approaches that leverage model uncertainty. We benchmark our method against state-of-the-art baselines from both categories, alongside two standard heuristics: random selection and Farthest Point Sampling (FPS). i. 2D-based: We adopt three no-reference image quality assessment (IQA) metrics, MUSIQ [29], MANIQA [68], and TOPIQ [9], implemented via the PyIQA [8] toolbox. ii. 3D-based: We compare against the prevailing uncertainty-driven baseline, FisherRF [24]. All baselines are re-implemented upon the radiative GS framework.

Implementation Details. Our pipeline is implemented in PyTorch [51] and optimized via Adam [31]. We set the ensemble size N = 10, perturbation parameters $\alpha = 1 0$ and $\beta = 0 . 5$ , and adopt the learning rates and loss weights from $R ^ { 2 } .$ -Gaussian [73]. All experiments are conducted on a single NVIDIA A40 GPU. For quantitative evaluation, we report the Peak Signal-to-Noise Ratio (PSNR) computed over the full 3D volume, and the Structural Similarity Index Measure (SSIM) [62] averaged across 2D slices in the axial, coronal, and sagittal planes.

<table><tr><td rowspan="2">Method</td><td colspan="2">24-view</td><td colspan="2">36-view</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>PSNRâ</td><td>SSIMâ</td></tr><tr><td>Random</td><td>41.862</td><td>0.954</td><td>44.552</td><td>0.961</td></tr><tr><td>FPS</td><td>43.081</td><td>0.957</td><td>45.386</td><td>0.964</td></tr><tr><td>TOPIQ [9]</td><td>42.898</td><td>0.961</td><td>45.483</td><td>0.967</td></tr><tr><td>MANIQA [68]</td><td>41.482</td><td>0.953</td><td>44.516</td><td>0.962</td></tr><tr><td>MUSIQ [29]</td><td>42.675</td><td>0.957</td><td>45.133</td><td>0.965</td></tr><tr><td>FisherRF [24]</td><td>43.289</td><td>0.962</td><td>46.212</td><td>0.969</td></tr><tr><td>Ours</td><td>44.069</td><td>0.966</td><td>46.849</td><td>0.971</td></tr></table>

Table 2: Quantitative comparison of novel view synthesis on the synthetic dataset using two different protocols. The best and second-best results are highlighted.

## 4.2 Results

Tomographic Reconstruction. Table 1 presents the quantitative results of 3D tomographic reconstruction across various active view selection strategies. Our approach demonstrates superior performance in both synthetic and realworld scenarios. In terms of 3D PSNR, our method surpasses the second-best baseline by margins of up to 0.68 dB on synthetic data and 0.22 dB on realworld data. Notably, FisherRF underperforms FPS under the synthetic 24-view setting. This degradation stems from FisherRFâs diagonal approximation of the Fisher Information Matrix. In X-ray imaging, strongly coupled Gaussians along projection rays violate this assumption, yielding severely biased information gain estimates. Qualitatively, as illustrated in Fig. 3, our method yields the most visually accurate reconstructions, faithfully preserving fine structural details, significantly suppressing boundary artifacts and background noise.

Novel View Synthesis. We also evaluate our methodâs performance on novel view synthesis using the synthetic dataset. The qualitative comparisons and quantitative results are reported in Fig. 4 and Tab. 2, respectively. Our approach achieves the best rendering quality under both protocols, demonstrating an improvement of up to 0.78 dB in PSNR. Visual comparisons indicate that our method better preserves local structural details, particularly in high-density areas. This improvement stems from our methodâs ability to acquire more informative viewpoints, which effectively mitigates the needle-like artifacts and noise that typically appear along rays passing through dense regions, e.g., the bones.

<!-- image-->  
Fig. 4: Visual comparisons of novel view synthesis results across different view selection strategies. The PSNRâ (dB) for each scene is displayed at the top-left corner of each image. Our approach achieves the highest rendering quality.

## 4.3 Ablation Study and Analysis

Component Analysis. To demonstrate the impact of our proposed structural variance-based view selection mechanism, we replace SSIM with L1 error and PSNR, respectively, and evaluate the performance under the 24-view protocol. As shown in the first section of Tab. 3, switching to L1 error or PSNR as the uncertainty metric leads to a significant performance drop. There are two primary reasons for this degradation. Firstly, L1 error and PSNR are pixel-wise, independently computed metrics based on absolute or mean-squared errors. They overlook the spatial arrangement and inter-dependencies of adjacent pixels, making them less sensitive to high-frequency topological changes, such as the structural discontinuities caused by perturbing needle-like artifacts. Secondly, due to the linear integral nature of X-ray imaging, perturbing the density of Gaussian primitives inevitably causes overall intensity shifts in the projection space. L1 and PSNR are highly susceptible to these absolute luminance fluctuations, which can easily overwhelm the subtle variance signals originating from genuine geometric ambiguities. In contrast, SSIM inherently incorporates luminance and contrast normalization, effectively decoupling absolute intensity shifts from structural information. This enables the SSIM variance to act as a more robust and precise indicator of true epistemic uncertainty and geometric disagreement.

Parameter Analysis. We evaluate the impact of ensemble size (N), ratio of perturbed Gaussian primitives (Î±), and density scaling amplitude (Î²) on reconstruction quality, as detailed in the last three sections of Tab. 3. Empirically, we observe that an ensemble size of $N = 1 0$ yields the optimal trade-off. Increasing N to 20 leads to a performance degradation. We attribute this to the oversmoothing of the structural disagreement signal: A larger sample size dilutes the impact of extreme catastrophic structural failures caused by perturbations, thereby reducing the discriminative contrast of the SSIM variance across views. A moderate size N successfully preserves the sharpness of the uncertainty landscape, allowing the active agent to accurately penalize geometric degeneracies.

The ratio Î± controls the proportion of Gaussians subjected to stochastic scaling, with 10% yields the best results. Both excessively low and high ratios deteriorate the performance. When Î± is too small, the perturbation is overly restricted to background noise or void space, failing with the primitives that constitute geometric degeneracies, resulting in a flat structural disagreement landscape. Conversely, a higher ratio indiscriminately extends the perturbations into well-constrained, high-confidence anatomical solids (e.g., bones). Due to the linear integral nature of X-ray projection, modulating these high-density structures induces massive, non-informative global variations that mask the subtle uncertainty signals from ambiguous regions.

For the scaling amplitude Î², where the density scaling factor is sampled from Uniform(âÎ², Î²), both overly conservative and excessively aggressive perturbations degrade the final reconstruction quality. A small Î² injects insufficient noise to disrupt the fragile geometric equilibrium of overfitting artifacts, providing weak guidance for view selection. Conversely, an excessively large Î² introduces severe out-of-distribution intensity shifts that completely distort local geometry. This structural collapse causes ubiquitous SSIM degradation across all views, diminishing the relative variance contrast and blinding the acquisition function.

## 5 Conclusion

We present a novel active view selection framework for sparse-view CT re-

<table><tr><td>Setting</td><td>PSNRâ</td><td>SSIMâ</td></tr><tr><td>L1</td><td>33.644</td><td>0.894</td></tr><tr><td>PSNR</td><td>33.390</td><td>0.889</td></tr><tr><td>SSIMâ </td><td>34.078</td><td>0.896</td></tr><tr><td>N = 1</td><td>33.687</td><td>0.887</td></tr><tr><td>N = 5</td><td>33.952</td><td>0.895</td></tr><tr><td>N = 10â </td><td>34.078</td><td>0.896</td></tr><tr><td>N = 20</td><td>33.865</td><td>0.895</td></tr><tr><td>Î± = 5%</td><td>33.680</td><td>0.893</td></tr><tr><td> $\alpha = 1 0 \% ^ { \dagger }$ </td><td>34.078</td><td>0.896</td></tr><tr><td> $\alpha = 1 5 \%$ </td><td>33.938</td><td>0.896</td></tr><tr><td>Î± = 20%</td><td>33.589</td><td>0.892</td></tr><tr><td> $\overline { { \beta = 0 . 1 } }$ </td><td>34.040</td><td>0.896</td></tr><tr><td> $\beta = 0 . 3$ </td><td>34.057</td><td>0.897</td></tr><tr><td> $\beta = 0 . 5 ^ { \dagger }$ </td><td>34.078</td><td>0.896</td></tr><tr><td> $\beta = 1 . 0$ </td><td>33.338</td><td>0.890</td></tr></table>

Table 3: Ablation study results on the uncertainty quantification metric, ensemble size (N ), perturbed ratio (Î±), and density scaling amplitude (Î²). Within each section, the best and second-best results are highlighted. â  indicates the default setting.

construction utilizing radiative Gaussian Splatting. To overcome the theoretical flaws of existing active learning in X-ray imaging, we introduce a physics-aware active view selection method based on Perturbed Gaussian Ensemble. By injecting stochastic density perturbations into under-constrained, low-density primitives, our approach exposes structural vulnerabilities quantified via projectionspace SSIM variance. This mechanism successfully targets and suppresses artifacts. Extensive evaluations confirm that our method yields superior volumetric reconstructions compared to existing baselines. This work bridges the gap between active learning and explicit radiative fields, advancing the practical deployment of 3DGS in dose-sensitive clinical and industrial settings.

## Acknowledgments

This research is supported by the National Artificial Intelligence Research Resource Pilot under award NAIRR250199. Computational resources are also provided by Delta and DeltaAI at the National Center for Supercomputing Applications through ACCESS allocations CIS250012, CIS250816, and CIS251188.

## References

1. Adler, J., Ãktem, O.: Learned primal-dual reconstruction. Transactions on Medical Imaging 37(6), 1322â1332 (2018)

2. Andersen, A.H., Kak, A.C.: Simultaneous algebraic reconstruction technique (sart): a superior implementation of the art algorithm. Ultrasonic imaging 6(1), 81â94 (1984)

3. Anirudh, R., Kim, H., Thiagarajan, J.J., Mohan, K.A., Champley, K., Bremer, T.: Lose the views: Limited angle ct reconstruction via implicit sinogram completion. In: CVPR. pp. 6343â6352 (2018)

4. Armato III, S.G., McLennan, G., Bidaut, L., McNitt-Gray, M.F., Meyer, C.R., Reeves, A.P., Zhao, B., Aberle, D.R., Henschke, C.I., Hoffman, E.A., et al.: The lung image database consortium (lidc) and image database resource initiative (idri): a completed reference database of lung nodules on ct scans. Medical physics 38(2), 915â931 (2011)

5. Cai, Y., Liang, Y., Wang, J., Wang, A., Zhang, Y., Yang, X., Zhou, Z., Yuille, A.: Radiative gaussian splatting for efficient x-ray novel view synthesis. In: European Conference on Computer Vision. pp. 283â299. Springer (2024)

6. Cai, Y., Wang, J., Yuille, A., Zhou, Z., Wang, A.: Structure-aware sparse-view xray 3d reconstruction. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 11174â11183 (2024)

7. Cai, Y., Xiao, Z., Liang, Y., Qin, M., Zhang, Y., Yang, X., Liu, Y., Yuille, A.L.: Hdrgs: Efficient high dynamic range novel view synthesis at 1000x speed via gaussian splatting. In: NeurIPS. pp. 68453â68471 (2024)

8. Chen, C., Mo, J.: IQA-PyTorch: Pytorch toolbox for image quality assessment. [Online]. Available: https://github.com/chaofengc/IQA-PyTorch (2022)

9. Chen, C., Mo, J., Hou, J., Wu, H., Liao, L., Sun, W., Yan, Q., Lin, W.: Topiq: A top-down approach from semantics to distortions for image quality assessment. TIP 33, 2404â2418 (2024)

10. Chen, L., Zhan, H., Chen, K., Xu, X., Yan, Q., Cai, C., Xu, Y.: Activegamer: Active gaussian mapping through efficient rendering. In: CVPR. pp. 16486â16497 (2025)

11. Chen, X., Li, Q., Wang, T., Xue, T., Pang, J.: Gennbv: Generalizable next-bestview policy for active 3d reconstruction. In: CVPR. pp. 16436â16445 (2024)

12. Chung, H., Ryu, D., McCann, M.T., Klasky, M.L., Ye, J.C.: Solving 3d inverse problems using pre-trained 2d diffusion models. In: CVPR. pp. 22542â22551 (2023)

13. Connolly, C.: The determination of next best views. In: ICRA. vol. 2, pp. 432â435. IEEE (1985)

14. Dhami, H., Sharma, V.D., Tokekar, P.: Pred-nbv: Prediction-guided next-best-view planning for 3d object reconstruction. In: IROS. pp. 7149â7154. IEEE (2023)

15. Duan, R., Chen, J., Kortylewski, A., Yuille, A., Liu, Y.: Prompt-based exemplar super-compression and regeneration for class-incremental learning. In: BMVC (2025)

16. Feldkamp, L.A., Davis, L.C., Kress, J.W.: Practical cone-beam algorithm. Journal of the Optical Society of America A 1(6), 612â619 (1984)

17. Feng, Z., Zhan, H., Chen, Z., Yan, Q., Xu, X., Cai, C., Li, B., Zhu, Q., Xu, Y.: Naruto: Neural active reconstruction from uncertain target observations. In: CVPR. pp. 21572â21583 (2024)

18. Fischer, T., Liu, Y., Jesslen, A., Ahmed, N., Kaushik, P., Wang, A., Yuille, A.L., Kortylewski, A., Ilg, E.: inemo: Incremental neural mesh models for robust classincremental learning. In: ECCV. pp. 357â374 (2024)

19. Gao, Z., Planche, B., Zheng, M., Chen, X., Chen, T., Wu, Z.: Ddgs-ct: Directiondisentangled gaussian splatting for realistic volume rendering. NeurIPS 37, 39281â 39302 (2024)

20. Ghani, M.U., Karl, W.C.: Deep learning-based sinogram completion for low-dose ct. In: IVMSP. pp. 1â5. IEEE (2018)

21. Gopalakrishnan, V., Golland, P.: Fast auto-differentiable digitally reconstructed radiographs for solving inverse problems in intraoperative imaging. In: Workshop on Clinical Image-Based Procedures. pp. 1â11. Springer (2022)

22. Huang, H., Wu, Y., Deng, C., Gao, G., Gu, M., Liu, Y.S.: Fatesgs: Fast and accurate sparse-view surface reconstruction using gaussian splatting with depth-feature consistency. In: AAAI. pp. 3644â3652 (2025)

23. Huang, H., Wu, Y., Zhou, J., Gao, G., Gu, M., Liu, Y.S.: Neusurf: On-surface priors for neural surface reconstruction from sparse input views. In: AAAI. pp. 2312â2320 (2024)

24. Jiang, W., Lei, B., Daniilidis, K.: Fisherrf: Active view selection and mapping with radiance fields using fisher information. In: ECCV. pp. 422â440. Springer (2024)

25. Jin, K.H., McCann, M.T., Froustey, E., Unser, M.: Deep convolutional neural network for inverse problems in imaging. TIP 26(9), 4509â4522 (2017)

26. Jin, L., Chen, X., RÃ¼ckin, J., PopoviÄ, M.: Neu-nbv: Next best view planning using uncertainty estimation in image-based neural rendering. In: IROS. pp. 11305â11312 (2023)

27. Jin, L., Zhong, X., Pan, Y., Behley, J., Stachniss, C., PopoviÄ, M.: Activegs: Active scene reconstruction using gaussian splatting. IEEE Robotics and Automation Letters (2025)

28. Kak, A.C., Slaney, M.: Principles of computerized tomographic imaging. SIAM (2001)

29. Ke, J., Wang, Q., Wang, Y., Milanfar, P., Yang, F.: Musiq: Multi-scale image quality transformer. In: ICCV. pp. 5148â5157 (2021)

30. Kerbl, B., Kopanas, G., LeimkÃ¼hler, T., Drettakis, G., et al.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42(4), 139â1 (2023)

31. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014)

32. Klacansky, P.: Open scivis datasets (December 2017), https://klacansky.com/ open-scivis-datasets/

33. Lee, S., Chung, H., Park, M., Park, J., Ryu, W.S., Ye, J.C.: Improving 3d imaging with pre-trained perpendicular 2d diffusion models. In: ICCV. pp. 10710â10720 (2023)

34. Li, J., Zhang, J., Bai, X., Zheng, J., Ning, X., Zhou, J., Gu, L.: Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with global-local depth normalization. In: CVPR. pp. 20775â20785 (2024)

35. Li, Y., Fu, X., Li, H., Zhao, S., Jin, R., Zhou, S.K.: 3dgr-ct: Sparse-view ct reconstruction with a 3d gaussian representation. Medical Image Analysis 103, 103585 (2025)

36. Li, Y., Kuang, Z., Li, T., Hao, Q., Yan, Z., Zhou, G., Zhang, S.: Activesplat: Highfidelity scene reconstruction through active gaussian splatting. IEEE Robotics and Automation Letters (2025)

37. Lin, Y., Luo, Z., Zhao, W., Li, X.: Learning deep intensity field for extremely sparse-view cbct reconstruction. In: MICCAI. pp. 13â23. Springer (2023)

38. Lin, Y., Yang, J., Wang, H., Ding, X., Zhao, W., Li, X.: CË 2rv: Cross-regional and cross-view learning for sparse-view cbct reconstruction. In: CVPR. pp. 11205â 11214 (2024)

39. Liu, J., Anirudh, R., Thiagarajan, J.J., He, S., Mohan, K.A., Kamilov, U.S., Kim, H.: Dolce: A model-based probabilistic diffusion framework for limited-angle ct reconstruction. In: ICCV. pp. 10498â10508 (2023)

40. Liu, Y., Li, Y., Schiele, B., Sun, Q.: Online hyperparameter optimization for classincremental learning. In: AAAI. pp. 8906â8913 (2023)

41. Liu, Y., Li, Y., Schiele, B., Sun, Q.: Wakening past concepts without past data: Class-incremental learning from online placebos. In: WACV. pp. 2226â2235 (2024)

42. Liu, Y., Schiele, B., Sun, Q.: Adaptive aggregation networks for class-incremental learning. In: CVPR. pp. 2544â2553 (2021)

43. Liu, Y., Schiele, B., Sun, Q.: Rmm: Reinforced memory management for classincremental learning. In: NeurIPS. pp. 3478â3490 (2021)

44. Liu, Y., Schiele, B., Vedaldi, A., Rupprecht, C.: Continual detection transformer for incremental object detection. In: CVPR. pp. 23799â23808 (2023)

45. Liu, Y., Su, Y., Liu, A.A., Schiele, B., Sun, Q.: Mnemonics training: Multi-class incremental learning without forgetting. In: CVPR. pp. 12245â12254 (2020)

46. Liu, Z., Bicer, T., Kettimuthu, R., Gursoy, D., De Carlo, F., Foster, I.: Tomogan: low-dose synchrotron x-ray tomography with generative adversarial networks: discussion. Journal of the Optical Society of America A 37(3), 422â434 (2020)

47. Liu, Z., Zha, R., Zhao, H., Li, H., Cui, Z.: 4drgs: 4d radiative gaussian splatting for efficient 3d vessel reconstruction from sparse-view dynamic dsa images. In: International Conference on Information Processing in Medical Imaging. pp. 361â 374. Springer (2025)

48. Manglos, S.H., Gagne, G.M., Krol, A., Thomas, F.D., Narayanaswamy, R.: Transmission maximum-likelihood reconstruction with ordered subsets for cone beam ct. Physics in Medicine & Biology 40(7), 1225â1241 (1995)

49. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65(1), 99â106 (2021)

50. Pan, X., Lai, Z., Song, S., Huang, G.: Activenerf: Learning where to see with uncertainty estimation. In: ECCV. pp. 230â246. Springer (2022)

51. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style, highperformance deep learning library. Advances in neural information processing systems 32 (2019)

52. Ran, Y., Zeng, J., He, S., Chen, J., Li, L., Chen, Y., Lee, G., Ye, Q.: Neurar: Neural uncertainty for autonomous 3d reconstruction with implicit neural representations. IEEE Robotics and Automation Letters 8(2), 1125â1132 (2023)

53. Roth, H., Farag, A., Turkbey, E.B., Lu, L., Liu, J., Summers, R.M.: Data from pancreas-ct (2016)

54. RÃ¼ckert, D., Wang, Y., Li, R., Idoughi, R., Heidrich, W.: Neat: Neural adaptive tomography. TOG 41(4), 1â13 (2022)

55. Rudin, L.I., Osher, S., Fatemi, E.: Nonlinear total variation based noise removal algorithms. Physica D: nonlinear phenomena 60(1-4), 259â268 (1992)

56. Sauer, K., Bouman, C.: A local update strategy for iterative reconstruction from projections. IEEE Transactions on Signal Processing 41(2), 534â548 (2002)

57. Scott, W.R., Roth, G., Rivest, J.F.: View planning for automated threedimensional object reconstruction and inspection. CSUR 35(1), 64â96 (2003)

58. Shen, L., Pauly, J., Xing, L.: Nerp: implicit neural representation learning with prior embedding for sparsely sampled image reconstruction. TNNLS 35(1), 770â 782 (2022)

59. Sidky, E.Y., Pan, X.: Image reconstruction in circular cone-beam computed tomography by constrained, total-variation minimization. Physics in Medicine & Biology 53(17), 4777â4807 (2008)

60. Society, T.F.I.P.: X-ray tomographic datasets (2024), https://fips.fi/category/ open-datasets/x-ray-tomographic-datasets/

61. Verboven, P., Dequeker, B., He, J., Pieters, M., Pols, L., Tempelaere, A., Van Doorselaer, L., Van Cauteren, H., Verma, U., Xiao, H., et al.: www. x-plant. org-the ct database of plant organs. In: 6th Symposium on X-ray Computed Tomography: Inauguration of the KU Leuven XCT Core Facility, Location: Leuven, Belgium (2022)

62. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. TIP 13(4), 600â612 (2004)

63. Wilson, J., Almeida, M., Mahajan, S., Labrie, M., Ghaffari, M., Ghasemalizadeh, O., Sun, M., Kuo, C.H., Sen, A.: Pop-gs: Next best view in 3d-gaussian splatting with p-optimality. In: CVPR. pp. 3646â3655 (2025)

64. Wu, Y., Huang, H., Zhang, W., Deng, C., Gao, G., Gu, M., Liu, Y.S.: Sparis: Neural implicit surface reconstruction of indoor scenes from sparse views. In: AAAI. pp. 8514â8522 (2025)

65. Xiao, W., Cruz, R.S., Ahmedt-Aristizabal, D., Salvado, O., Fookes, C., Lebrat, L.: Nerf director: Revisiting view selection in neural volume rendering. In: CVPR. pp. 20742â20751 (2024)

66. Xie, Y., Cai, Y., Zhang, Y., Yang, L., Pan, J.: Gauss-mi: Gaussian splatting shannon mutual information for active 3d reconstruction. arXiv preprint arXiv:2504.21067 (2025)

67. Xue, S., Dill, J., Mathur, P., Dellaert, F., Tsiotra, P., Xu, D.: Neural visibility field for uncertainty-driven active mapping. In: CVPR. pp. 18122â18132 (2024)

68. Yang, S., Wu, T., Shi, S., Lao, S., Gong, Y., Cao, M., Wang, J., Yang, Y.: Maniqa: Multi-dimension attention network for no-reference image quality assessment. In: CVPR. pp. 1191â1200 (2022)

69. Ying, X., Guo, H., Ma, K., Wu, J., Weng, Z., Zheng, Y.: X2ct-gan: reconstructing ct from biplanar x-rays with generative adversarial networks. In: CVPR. pp. 10619â 10628 (2019)

70. Yu, L., Zou, Y., Sidky, E.Y., Pelizzari, C.A., Munro, P., Pan, X.: Region of interest reconstruction from truncated data in circular cone-beam ct. Transactions on Medical Imaging 25(7), 869â881 (2006)

71. Yu, W., Cai, Y., Zha, R., Fan, Z., Li, C., Yuan, Y.: X2-gaussian: 4d radiative gaussian splatting for continuous-time tomographic reconstruction. In: ICCV. pp. 24728â24738 (2025)

72. Zang, G., Idoughi, R., Li, R., Wonka, P., Heidrich, W.: Intratomo: self-supervised learning-based tomography via sinogram synthesis and prediction. In: ICCV. pp. 1960â1970 (2021)

73. Zha, R., Lin, T.J., Cai, Y., Cao, J., Zhang, Y., Li, H.: RË2-gaussian: Rectifying radiative gaussian splatting for tomographic reconstruction. In: Advances in Neural Information Processing Systems. vol. 37, pp. 44907â44934 (2024)

74. Zha, R., Zhang, Y., Li, H.: Naf: neural attenuation fields for sparse-view cbct reconstruction. In: MICCAI. pp. 442â452. Springer (2022)

75. Zhang, G., Zha, R., He, H., Liang, Y., Yuille, A., Li, H., Cai, Y.: X-lrm: X-ray large reconstruction model for extremely sparse-view computed tomography recovery in one second. arXiv preprint arXiv:2503.06382 (2025)

76. Zhang, J., Li, J., Yu, X., Huang, L., Gu, L., Zheng, J., Bai, X.: Cor-gs: sparseview 3d gaussian splatting via co-regularization. In: ECCV. pp. 335â352. Springer (2024)

77. Zhang, Y., Li, X., Chen, H., Yuille, A.L., Liu, Y., Zhou, Z.: Continual learning for abdominal multi-organ and tumor segmentation. In: MICCAI. pp. 35â45. Springer (2023)

78. Zhu, Z., Fan, Z., Jiang, Y., Wang, Z.: Fsgs: Real-time few-shot view synthesis using gaussian splatting. In: ECCV. pp. 145â163. Springer (2024)