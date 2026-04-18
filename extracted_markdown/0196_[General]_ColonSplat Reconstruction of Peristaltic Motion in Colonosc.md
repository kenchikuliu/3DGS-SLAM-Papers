# ColonSplat: Reconstruction of Peristaltic Motion in Colonoscopy with Dynamic Gaussian Splatting

Weronika Smolak-DyÅ¼ewska1â, Joanna Kaleta2,3â, Diego DallâAlba4,3, and PrzemysÅaw Spurek1,5

1 Jagiellonian University, KrakÃ³w, Poland

2 Warsaw University of Technology, Poland

3 Sano Centre for Computational Medicine, KrakÃ³w, Poland

4 University of Verona, Italy,

5 IDEAS Research Institute, Warsaw, Poland weronika.smolak@doctoral.uj.edu.pl

Abstract. Accurate 3D reconstruction of colonoscopy data, accounting for complex peristaltic movements, is crucial for advanced surgical navigation and retrospective diagnostics. While recent novel view synthesis and 3D reconstruction methods have demonstrated remarkable success in general endoscopic scenarios, they struggle in the highly constrained environment of the colon. Due to the limited field of view of a camera moving through an actively deforming tubular structure, existing endoscopic methods reconstruct the colon appearance only for initial camera trajectory. However, the underlying anatomy remains largely static; instead of updating Gaussiansâ spatial coordinates (xyz), these methods encode deformation through either rotation, scale or opacity adjustments. In this paper, we first present a benchmark analysis of state-of-the-art dynamic endoscopic methods for realistic colonoscopic scenes, showing that they fail to model true anatomical motion. To enable rigorous evaluation of global reconstruction quality, we introduce DynamicColon, a synthetic dataset with ground-truth point clouds at every timestep. Building on these insights, we propose ColonSplat, a dynamic Gaussian Splatting framework that captures peristaltic-like motion while preserving global geometric consistency, achieving superior geometric fidelity on C3VDv2 and DynamicColon datasets. Project page: https://wmito.github.io/ColonSplat

Keywords: Dynamic 3DGS Â· colonoscopy Â· 3D reconstruction

## 1 Introduction

Endoluminal endoscopic procedures are central to clinical diagnosis, with colonoscopy serving as the gold standard for colorectal cancer screening. Accurate 3D reconstruction of the colon is a fundamental challenge in medical image analysis, essential for advanced endoscopic navigation, precise polyp localization, and retrospective diagnostics. The recent emergence of novel view synthesis techniques, particularly Neural Radiance Fields (NeRF) [14] and 3D Gaussian Splatting (3DGS) [11], has revolutionized the field of computational anatomy by enabling high-fidelity rendering of complex anatomy. Based on these foundational representations, numerous works have attempted to reconstruct colonoscopy scenes [1, 2, 6, 10, 16, 19]. However, a common limitation of these colon-specific methods is their reliance on a non-realistic static synthetic datasets. By treating the colon as a rigid structure, they ignore the complex, continuous peristaltic movements that define in vivo examinations, limiting their clinical applicability.

<!-- image-->  
Fig. 1. The C3VDv2 dataset contains realistic colonoscopy sequences with substantial non-rigid deformations over time. The top two rows show captures from the endoscopic camera and the corresponding renders produced by different methods. The bottom row illustrates the global structure of the reconstructed colon. Peristaltic-like dynamic challenge baseline approaches; however, ColonSplat uniquely maintains a physically plausible global structure across timesteps. Please zoom in for details, see more examples in supplementary videos.

Parallel to these efforts, significant progress has been made in dynamic endoscopic reconstruction, successfully modeling tissue deformations based on both NeRF [20, 23] and 3DGS [4, 8, 9, 12, 13, 15, 18]. While these dynamic methods achieve impressive results, they are primarily designed for general endoscopic or laparoscopic procedures. In contrast to wide-cavity scenarios with localized movement, colonoscopy navigates a constrained, narrow tube. Here, constant camera translation and non-rigid peristalsis deprive the system of persistent landmarks. Existing methods struggle with this deforming tubular topology and restricted field of view. Due to these limitations, current models only succeed at capturing local tissue appearance consistent with views from initial camera trajectory. Global evaluation reveals geometric inconsistencies in the reconstructed colonic structure, clearly visible when looking from camera located outside of the colon (see Fig. 1). Enforcing physically plausible deformations is important because: (i) overfitted Gaussians that explain motion through rotations and scale changes may lead to artifacts in novel views deviating from the training trajectory, and (ii) physically grounded reconstructions provide a more reliable foundation for integration with potential downstream physics-based simulations.

To address this crucial gap, we introduce ColonSplat, a novel dynamic 3DGS approach tailored specifically for the extreme conditions of colonoscopy. ColonSplat is designed to accurately capture the complex deformations of the tissue while explicitly maintaining the global structural integrity of the reconstructed tubular organ. Furthermore, a major obstacle in evaluating global reconstruction accuracy in colonoscopy is the lack of clinical datasets with reliable 3D ground truth of geometry. To rigorously benchmark the performance of global reconstruction, a task impossible with clinical data alone, we present a novel synthetic dataset of a dynamically moving colon with precise global trajectories and geometry. In summary, our main contributions are as follows:

â Benchmark Analysis: We provide a thorough evaluation of state-of-theart dynamic endoscopic methods in the context of colonoscopy. Our analysis shows that these approaches fail to model true anatomical motion; instead of meaningfully updating Gaussiansâ spatial coordinates (xyz), they encode deformation through rotation, scale and opacity, resulting in physically implausible motion reconstructions.

â DynamicColon Dataset: We introduce new dynamic dataset that provides ground-truth point clouds for every timestep. Unlike existing datasets, it enables evaluation of global geometric reconstruction and structural integrity in the presence of tissue deformations.

â ColonSplat Method: We propose novel dynamic reconstruction framework that successfully captures complex peristaltic-like motion while preserving global geometric consistency, achieving superior geometric fidelity on both the highly realistic C3VDv2 [5] and DynamicColon datasets.

## 2 Related Work

Static Novel View Synthesis in Colonoscopy. The introduction of NeRF [14] and 3DGS [11] has significantly advanced the capabilities of novel view synthesis in medical imaging. In the specific context of colonoscopy, several methods have been developed to reconstruct the complex geometry and texture of the colon interior. NeRF-based approaches, such as REIM-NERF [16] and NFL-BA [1], utilize implicit neural representations to model the tissue. More recently, 3DGS has been adapted for colonoscopy to leverage its explicit representation and real-time rendering capabilities, as seen in Gaussian Pancakes [2], EndoGSLAM [19], PR-ENDO [10], and EndoPBR [6]. While these methods achieve high fidelity reconstructions, they fundamentally assume an unnaturally static environment. By modeling the colon as a rigid structure, they cannot account for the continuous peristaltic motion observed during in vivo procedures, thereby limiting their applicability in realistic dynamic scenarios.

Dynamic Reconstruction in General Endoscopy. To address tissue deformation during surgery, various dynamic reconstruction techniques have been proposed, primarily focusing on general endoscopic and laparoscopic procedures.

<!-- image-->  
Fig. 2. ColonSplat reconstructs dynamic 3D anatomy from colonoscopy video using estimated depth. A deformation model updates canonical Gaussians parameters at each time step for consistent dynamic reconstruction. Training uses RGB and depth supervision, with KNN and color regularization, to ensure accurate, artifact-free results.

Early dynamic methods relied on NeRF architectures [20, 23] to model non-rigid deformations over time. However, the implicit nature of these models often results in computationally expensive training and rendering, making it difficult to explicitly track tissue movement in real-time clinical applications.

A multitude of recent works, including EndoPlanar [15], Endo-4DGS [9], ENDO-4DGX [8], SurgicalGS [4], SGS [18], EndoGaussian [13], and EndoSparse [12], have demonstrated exceptional performance in modeling surgical scenes using dynamic 3DGS-based frameworks. While these methods effectively capture dynamic tool-tissue interactions, they are mainly designed for cavity-like environments where the region of interest remains consistently visible. In such laparoscopic settings, deformations typically unfold within a broad field of view that provides sufficient geometric overlap for spatial consistency.

Addressing the Challenges of Dynamic Colonoscopy. Despite the success of dynamic 3DGS in laparoscopy, directly applying these state-of-the-art methods to dynamic colonoscopy reveals significant limitations. Colonoscopy presents a unique set of challenges characterized by a highly constrained tubular topology, a limited field of view, and global peristaltic motion. As the camera continuously translates forward, previously observed regions quickly leave the field of view.

Consequently, when existing dynamic endoscopic methods are deployed in this environment, they successfully reconstruct the tissue appearance on a purely local scale but fail to maintain global geometric consistency. The lack of persistent global landmarks, combined with continuous non-rigid deformation and forward camera translation, leads to severe geometric inconsistencies and profound structural artifacts in these models, especially when viewed globally. Our proposed method, ColonSplat, focuses on reconstructing movement of a colon as global changes to geometry rather than local overfitting to initial camera trajectory.

## 3 Method

This section presents the components of our method, summarized in Fig. 2.

Gaussian Splatting Representation. We represent the scene as a collection of 3D anisotropic Gaussians rendered using differentiable rasterization. Each Gaussian in canonical space is parameterized by a mean position $\mathbf { x } \in \mathbb { R } ^ { 3 }$ , scale $\textbf { s } \in \mathbb { R } ^ { 3 }$ , rotation $\textbf { r } \in \mathbb { R } ^ { 4 }$ , opacity $\alpha \in \mathbb { R } ^ { 1 }$ , and color $\mathbf { c } \in \mathbb { R } ^ { 3 }$ . Additionally, each Gaussian has a learnable embedding vector $\mathbf { e } _ { i } \in \mathbb { R } ^ { d }$ used exclusively for color prediction. Given calibrated RGB images and camera poses, we use a monocular depth estimator (ColonCrafter [7] or AnyDepth [17]) to predict depth maps, from which we initialize the Gaussian point cloud. To render novel views Gaussians are projected to the image plane and alpha-composited in depth order.

Dynamic Deformation Modeling. Colon motion is modeled with a multilayer perceptron Î¸ that predicts time-dependent geometric updates. Given timestep $t \in [ 0 , 1 ]$ and canonical Gaussian parameters, we predict geometric offsets:

$$
( \varDelta \mathbf { x } , \varDelta \mathbf { s } , \varDelta \mathbf { r } ) = \theta \big ( \mathop { \mathrm { G } ( t ) , \mathrm { G } ( \mathbf { x } ) } \big ) .\tag{1}
$$

G denotes encoding with HexPlane Grid, following [3, 21]. The deformed geometric parameters are obtained additively:

$$
\mathbf { x } ^ { \prime } = \mathbf { x } + \varDelta \mathbf { x } , \quad \mathbf { s } ^ { \prime } = \mathbf { s } + \mathrm { c l i p } ( \varDelta \mathbf { s } , - \tau _ { s } , \tau _ { s } ) , \quad \mathbf { r } ^ { \prime } = \mathbf { r } + \mathrm { c l i p } ( \varDelta \mathbf { r } , - \tau _ { r } , \tau _ { r } ) .\tag{2}
$$

Clipping thresholds $\tau _ { s }$ and $\tau _ { r }$ restrict the magnitude of scale and rotation updates, preventing these parameters from compensating for insufficient spatial deformation. Opacity Î± is fixed over time to prevent trivially explaining motion by hiding and revealing Gaussians. Additionally, updated Gaussian scales $s ^ { \prime }$ are limited to 5% of the scene extent to prevent oversized primitives.

Color Modeling. Endoscopic imaging exhibits strong appearance variations due to the collocated camera and light source, as well as fluid-related effects in realistic colon environments. Moreover, local tissue characteristics (e.g., fluids or polyps) may exhibit distinct temporal reflectance behavior. To account for these illumination and tissue-specific effects, color is modeled separately using a dedicated branch of the network, denoted as $\theta _ { \mathrm { c o l o r } }$ , which leverages the per-Gaussian embedding $\mathbf { e } _ { i }$ to capture localized temporal appearance variations. Color updates are modeled multiplicatively to capture light-surface interactions:

$$
\begin{array} { r } { \Delta \mathbf { c } = \theta _ { \mathrm { c o l o r } } \big ( \mathbf { G } ( t ) , \mathbf { G } ( \mathbf { x } ) , \mathbf { e } _ { i } \big ) , \quad \mathbf { c } ^ { \prime } = \mathbf { c } \odot ( 1 + \Delta \mathbf { c } ) . } \end{array}\tag{3}
$$

Following [4], we optimize $\ell _ { 1 }$ RGB reconstruction loss $\mathcal { L } _ { \mathrm { r g b } }$ between the rendered and GT images, and apply a total variation loss ${ \mathcal { L } } _ { \mathrm { T V } }$ on the renders to promote spatial smoothness and reduce high-frequency artifacts.

K-Nearest Neighbor Deformation Consistency. To encourage locally coherent non-rigid motion, for each Gaussian i we identify its K nearest neighbors ${ \mathcal { N } } _ { i }$ in canonical space and penalize inconsistencies in their deformed positions:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { k n n } } = \frac { 1 } { N } \sum _ { i } \left. \mathbf { x } _ { i } ^ { \prime } ( t ) - \frac { 1 } { | \mathcal { N } _ { i } | } \sum _ { j \in \mathcal { N } _ { i } } \mathbf { x } _ { j } ^ { \prime } ( t ) \right. _ { 2 } ^ { 2 } . } \end{array}\tag{4}
$$

Color regularization. To prevent motion from being spuriously explained through color variation, we penalize predicted color offsets $\left( { { L _ { c o } } } \right)$ . Additional ${ \mathrm { l y } } ,$ we slighlty penalize color variance across all Gaussians to avoid implausible color artifacts when only partial geometry is observed $\left( { { L _ { c v } } } \right)$

$$
\begin{array} { r l r } { \mathcal { L } _ { \mathrm { c o } } = \frac { 1 } { N } \sum _ { i } \left\| \Delta \mathbf { c } _ { i } \right\| _ { 2 } ^ { 2 } , } & { } & { \mathcal { L } _ { \mathrm { c v } } = \frac { 1 } { N } \sum _ { i } \left\| \mathbf { c } _ { i } ^ { \prime } - \frac { 1 } { N } \sum _ { j } \mathbf { c } _ { j } ^ { \prime } \right\| _ { 2 } ^ { 2 } . } \end{array}\tag{5}
$$

Depth regularization. Geometry is supervised using monocular depth estimates by applying an $\ell _ { 1 }$ loss between normalized supervision depth and rendered expected depth:

$$
\mathcal { L } _ { \mathrm { d e p t h } } = \left. D _ { \mathrm { r e n d e r e d } } ^ { \mathrm { n o r m } } - D _ { \mathrm { s u p } } ^ { \mathrm { n o r m } } \right. _ { 1 } .\tag{6}
$$

Training Objective. The objective combines reconstruction and regularizers:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { r g b } } + \lambda _ { \mathrm { T V } } \mathcal { L } _ { \mathrm { T V } } + \lambda _ { \mathrm { k n n } } \mathcal { L } _ { \mathrm { k n n } } + \lambda _ { \mathrm { d e p t h } } \mathcal { L } _ { \mathrm { d e p t h } } + \lambda _ { \mathrm { c o } } \mathcal { L } _ { \mathrm { c o } } + \lambda _ { \mathrm { c v } } \mathcal { L } _ { \mathrm { c v } } .\tag{7}
$$

## 4 Experiments

We provide data, code and all implementation details in our GitHub repository6.

Datasets. Our evaluation utilizes the C3VDv2 dataset [5] for its high fidelity. It introduces challenging visual artifacts like blood and mucus, while its simulated peristaltic-like motion creates the significant global geometric deformations similar to live colonoscopies. These characteristics make reconstruction significantly more challenging than previously studied endoscopic benchmarks. We used 9 sequences from C3VDv2 with deformation for which ground truth camera poses were available. Following standard practice, we use every 8th frame for testing and the remaining frames for training.

Additionally, we introduce DynamicColon dataset created from C3VDv2 meshes, custom textures and cage based colon deformations. It consists of three scenes with train and test camera trajectories, camera views from outside, depth maps and point clouds for every deformation step which enables precise evaluation of geometric and deformation fidelity. Dataset is provided on our repository.

Baselines. We compare ColonSplat with several recent dynamic endoscopic reconstruction methods: Endo4DGS [9], Deform3DGS [22], SurgicalGS [4], EndoPlanar [15], Endo4DGX [8]. As these methods handle depth differently - some relying on normalized depth, others on metric depth - their original depth losses are not always directly compatible with ColonCrafter depth maps. This in practice led to convergence issues. To ensure a fair comparison, we apply the same L1 depth loss on normalized depth maps across all methods. We also tune depth loss weights and training iterations for each baseline to obtain the most satisfactory renderings and consistent geometry for each method to ensure fair comparison.

Table 1. Quantitative comparison and ablation study on C3VDv2 and DynamicColon. Compared to baselines ColonSplat achieve superior geometry fidelity while offering strong reconstruction results. Ablation is shown in three bottom rows. \*Lack of âc modeling caused major Gaussian drift and degraded CH and HD95 in one scene.
<table><tr><td></td><td colspan="3">C3VDv2</td><td colspan="6">DynamicColon</td></tr><tr><td>Method</td><td>PSNR â SSIM âLPIPS â|PSNR â SSIM âLPIPS â CH â HD95 â MSED â</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EndoPlanar</td><td>27.229</td><td>0.738</td><td>0.499</td><td>28.195</td><td>0.902</td><td>0.332</td><td>0.497</td><td>1.055</td><td>0.2999</td></tr><tr><td>SurgicalGS</td><td>27.624</td><td>0.766</td><td>0.350</td><td>31.635</td><td>0.942</td><td>0.075</td><td>0.384</td><td>1.136</td><td>0.0079</td></tr><tr><td>Deform3DGS</td><td>27.419</td><td>0.745</td><td>0.395</td><td>31.635</td><td>0.942</td><td>0.109</td><td>0.383</td><td>1.143</td><td>0.0077</td></tr><tr><td>Endo4DGS</td><td>27.491</td><td>0.728</td><td>0.582</td><td>30.548</td><td>0.945</td><td>0.155</td><td>0.372</td><td>1.017</td><td>0.0080</td></tr><tr><td>Endo4DGX</td><td>24.525</td><td>0.707</td><td>0.521</td><td>29.722</td><td>0.921</td><td>0.322</td><td>0.433</td><td>1.133</td><td>0.0169</td></tr></table>

<table><tr><td>ColonSplat</td><td>28.281</td><td>0.750</td><td>0.444</td><td>33.633</td><td>0.955</td><td>0.137</td><td>0.162</td><td>0.730</td><td>0.0060</td></tr><tr><td>w/o Constraints</td><td>27.624</td><td>0.736</td><td>0.523</td><td>34.536</td><td>0.959</td><td>0.126</td><td>0.234</td><td>0.854</td><td>0.0055</td></tr><tr><td>w/o LKNN</td><td>28.452</td><td>0.752</td><td>0.438</td><td>30.548</td><td>0.945</td><td>0.145</td><td>0.981</td><td>1.033</td><td>0.0060</td></tr><tr><td>w/o âc modelling</td><td>27.579</td><td>0.733</td><td>0.496</td><td>25.857</td><td>0.843</td><td>0.342</td><td>11.207*</td><td>3.544*</td><td>0.0305</td></tr></table>

<!-- image-->  
Fig. 3. Qualitative comparison for C3VDv2 dataset. Please zoom in for details. ColonSplat achieves superior reconstruction quality compared to the baselines.

Quantitative Comparison. Tab. 1 reports quantitative results across all baselines and ColonSplat. For C3VDv2, we report standard image reconstruction metrics computed on held-out test frames: PSNR, SSIM, and LPIPS.

ColonSplat achieves strong reconstruction quality compared to baselines. For the DynamicColon with available ground-truth geometry, we additionally report Chamfer Distance (CD) and 95th percentile Hausdorff Distance (HD95) on a point cloud sampled from Gaussian representation at each test timestep, and MSE on normalized render depths $\mathrm { M S E } _ { D }$ . These metrics enable direct evaluation of deformation fidelity. ColonSplat consistently improves geometric consistency under strong peristaltic-like deformation.

<!-- image-->

Fig. 4. DynamicColon. The top row shows renders from test trajectory. The bottom row presents views from cameras positioned outside the colon. Competing methods produce significant artifacts that mimic deformations. ColonSplat accurately captures the 3D structure of the colon without such artifacts, even when viewed from outside.  
<!-- image-->  
Fig. 5. Ablation study on C3VDv2 scene. Our proposed components significantly enhance realistic dynamic colon deformation. KNN regularization provides smooth tissuelike Gaussian surface across timesteps while our proposed constraints eliminate geometric artifacts. Please zoom in for details.

Qualitative Results. We present qualitative comparisons in Fig. 3 and 4. ColonSplat produces more stable large-scale geometry and fewer view-dependent artifacts compared to prior methods, particularly in regions affected by strong tissue deformation. We strongly encourage to view our supplement.

Ablation Study. To evaluate the contribution of individual components, we perform ablations under three settings: (i) No Constraints, where all Gaussian parameters are freely optimized without structural regularization; (ii) No KNN Consistency; and (iii) No Color Modeling (âc). Quantitative results are reported in Tab. 1 and qualitative results are shown in Fig. 5.

## 5 Conclusions

ColonSplat introduces a dynamic Gaussian Splatting framework tailored for colonoscopy that captures complex motion while preserving global anatomical consistency. By explicitly constraining deformation and incorporating additional supervision, it mitigates the structural artifacts observed in prior dynamic baselines. Extensive analysis of state-of-the-art baselines on C3VDv2 and the proposed DynamicColon dataset demonstrates superior geometric fidelity, strong reconstruction quality and provides evaluation framework for future works.

Limitations: Despite significant improvements, motion reconstruction can remain inaccurate when deformations are very poorly observed in the video.

Acknowledgements. Joanna Kaleta is supported by National Science Centre, Poland (grant no. 2022/47/O/ST6/01407). This paper received funding from the European Unionâs Horizon 2020 research and innovation programme under grant agreement No 857533. The research is supported by Sano project carried out within the International Research Agendas programme of the Foundation for Polish Science, co-financed by the European Union under the European Regional Development Fund. The research was created within the project of the Minister of Science and Higher Education âSupport for the activity of Centers of Excellence established in Poland under Horizon 2020â on the basis of the contract number MEiN/2023/DIR/3796. The work of W. Smolak-DyÅ¼ewska and P. Spurek was supported by the project Effective Rendering of 3D Objects Using Gaussian Splatting in an Augmented Reality Environment (FENG.02.02-IP.05- 0114/23), carried out under the First Team programme of the Foundation for Polish Science and co-financed by the European Union through the European Funds for Smart Economy 2021â2027 (FENG).

## References

1. Beltran, A.D., Rho, D., Niethammer, M., Sengupta, R.: Nfl-ba: Near-field light bundle adjustment for slam in dynamic lighting. arXiv preprint arXiv:2412.13176 (2024)

2. Bonilla, S., Zhang, S., Psychogyios, D., Stoyanov, D., Vasconcelos, F., Bano, S.: Gaussian pancakes: geometrically-regularized 3d gaussian splatting for realistic endoscopic reconstruction. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 274â283. Springer (2024)

3. Cao, A., Johnson, J.: Hexplane: A fast representation for dynamic scenes. CVPR (2023)

4. Chen, J., Zhang, X., Hoque, M.I., Vasconcelos, F., Stoyanov, D., Elson, D.S., Huang, B.: Surgicalgs: Dynamic 3d gaussian splatting for accurate robotic-assisted surgical scene reconstruction. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 572â582. Springer (2025)

5. Golhar, M.V., Fretes, L.S.G., Ayers, L., Akshintala, V.S., Bobrow, T.L., Durr, N.J.: C3vdv2 â colonoscopy 3d video dataset with enhanced realism (2025), https://arxiv.org/abs/2506.24074

6. Han, J.J., Wu, J.Y.: Endopbr: Material and lighting estimation for photorealistic surgical simulations via physically-based rendering. arXiv preprint arXiv:2502.20669 (2025)

7. Hardy, R., Berzin, T.M., Rajpurkar, P.: Coloncrafter: A depth estimation model for colonoscopy videos using diffusion priors. In: Biocomputing 2026: Proceedings of the Pacific Symposium. pp. 27â41. World Scientific (2025)

8. Huang, Y., Bai, L., Cui, B., Li, Y., Chen, T., Wang, J., Wu, J., Lei, Z., Liu, H., Ren, H.: Endo-4dgx: Robust endoscopic scene reconstruction and illumination correction with gaussian splatting. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 181â191. Springer (2025)

9. Huang, Y., Cui, B., Bai, L., Guo, Z., Xu, M., Islam, M., Ren, H.: Endo-4dgs: Endoscopic monocular scene reconstruction with 4d gaussian splatting. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 197â207. Springer (2024)

10. Kaleta, J., Smolak-DyÅ¼ewska, W., Malarz, D., DallâAlba, D., Korzeniowski, P., Spurek, P.: PR-ENDO: Physically Based Relightable Gaussian Splatting for Endoscopy . In: proceedings of Medical Image Computing and Computer Assisted Intervention â MICCAI 2025. vol. LNCS 15969. Springer Nature Switzerland (September 2025)

11. Kerbl, B., Kopanas, G., LeimkÃ¼hler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics 42(4) (2023)

12. Li, C., Feng, B.Y., Liu, Y., Liu, H., Wang, C., Yu, W., Yuan, Y.: Endosparse: Real-time sparse view synthesis of endoscopic scenes using gaussian splatting. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 252â262. Springer (2024)

13. Liu, Y., Li, C., Yang, C., Yuan, Y.: Endogaussian: Real-time gaussian splatting for dynamic endoscopic scene reconstruction. arXiv preprint arXiv:2401.12561 (2024)

14. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In: ECCV (2020)

15. Paonim, T., Sasnarukkit, C., Nupairoj, N., Vateekul, P.: Endoplanar: Deformable planar-based gaussian splatting for surgical scene reconstruction. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 127â136. Springer (2025)

16. Psychogyios, D., Vasconcelos, F., Stoyanov, D.: Realistic endoscopic illumination modeling for nerf-based data generation. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 535â544. Springer (2023)

17. Ren, Z., Zhang, Z., Li, W., Tang, H.: Anydepth: Depth estimation made easy (2026)

18. Sunmola, I.O., Zhao, Z., Schmidgall, S., Wang, Y., Scheikl, P.M., Pham, V., Krieger, A.: Surgical gaussian surfels: Highly accurate real-time surgical scene rendering using gaussian surfels. arXiv preprint arXiv:2503.04079 (2025)

19. Wang, K., Yang, C., Wang, Y., Li, S., Wang, Y., Dou, Q., Yang, X., Shen, W.: Endogslam: Real-time dense reconstruction and tracking in endoscopic surgeries using gaussian splatting. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 219â229. Springer (2024)

20. Wang, Y., Long, Y., Fan, S.H., Dou, Q.: Neural rendering for stereo 3d reconstruction of deformable tissues in robotic surgery. In: International conference on medical image computing and computer-assisted intervention. pp. 431â441. Springer (2022)

21. Wu, G., Yi, T., Fang, J., Xie, L., Zhang, X., Wei, W., Liu, W., Tian, Q., Wang, X.: 4d gaussian splatting for real-time dynamic scene rendering. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 20310â20320 (June 2024)

22. Yang, S., Li, Q., Shen, D., Gong, B., Dou, Q., Jin, Y.: Deform3DGS: Flexible Deformation for Fast Surgical Scene Reconstruction with Gaussian Splatting . In: proceedings of Medical Image Computing and Computer Assisted Intervention â MICCAI 2024. vol. LNCS 15006. Springer Nature Switzerland (October 2024)

23. Zha, R., Cheng, X., Li, H., Harandi, M., Ge, Z.: Endosurf: Neural surface reconstruction of deformable tissues with stereo endoscope videos. In: International conference on medical image computing and computer-assisted intervention. pp. 13â23. Springer (2023)