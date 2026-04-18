# ColorGS: High-fidelity Surgical Scene Reconstruction with Colored Gaussian Splatting

Qun Ji, Peng Li, Mingqiang Wei

Nanjing University of Aeronautics and Astronautics

Abstract. High-fidelity reconstruction of deformable tissues from endoscopic videos remains challenging due to the limitations of existing methods in capturing subtle color variations and modeling global deformations. While 3D Gaussian Splatting (3DGS) enables efficient dynamic reconstruction, its fixed per-Gaussian color assignment struggles with intricate textures, and linear deformation modeling fails to model consistent global deformation. To address these issues, we propose ColorGS, a novel framework that integrates spatially adaptive color encoding and enhanced deformation modeling for surgical scene reconstruction. First, we introduce Colored Gaussian Primitives, which employ dynamic anchors with learnable color parameters to adaptively encode spatially varying textures, significantly improving color expressiveness under complex lighting and tissue similarity. Second, we design an Enhanced Deformation Model (EDM) that combines time-aware Gaussian basis functions with learnable time-independent deformations, enabling precise capture of both localized tissue deformations and global motion consistency caused by surgical interactions. Extensive experiments on DaVinci robotic surgery videos and benchmark datasets (EndoNeRF, StereoMIS) demonstrate that ColorGS achieves state-of-the-art performance, attaining a PSNR of 39.85 (1.5 higher than prior 3DGS-based methods) and superior SSIM (97.25%) while maintaining real-time rendering efficiency. Our work advances surgical scene reconstruction by balancing high fidelity with computational practicality, critical for intraoperative guidance and AR/VR applications.

Keywords: 3D Reconstruction Â· Gaussian Splatting Â· Endoscopic Surgery.

## 1 Introduction

Reconstructing 3D surgical scenes from endoscopic videos is essential for intraoperative navigation [14], enhanced visualization [10,15], and robotic-assisted surgery (RAMIS) [24]. Beyond providing real-time guidance, it also supports AR/VR-based surgical training [17] and preoperative planning by providing high-fidelity anatomical models. Moreover, improved 3D reconstruction contributes to optimizing surgical workflows, facilitating remote monitoring [20], immersive visualization, and accelerated skill acquisition [1]. Thus, achieving accurate and efficient reconstruction is indispensable for advancing surgical practices and improving patient care.

Early methods for surgical scene reconstruction mainly rely on depth estimation [12] and SLAM-based point cloud fusion [16,24]. However, these methods often suffer from inefficiencies due to complex processing pipelines and redundant data integration, limiting both real-time performance and accuracy. Recent studies, such as EndoNeRF [18] and EndoSurf [23], have shifted towards using neural radiance fields (NeRF) [13], combined with differentiable rendering techniques to achieve high-quality and efficient surgical scene reconstruction. Despite the improved reconstruction quality and streamlined pipeline, the high computational cost of NeRF leads to long training times and slow rendering speeds, significantly hindering its applicability in real-time surgical scenarios.

The emergence of 3D Gaussian Splatting (3DGS) [6] marks a significant breakthrough, enabling high-fidelity reconstruction with greater efficiency compared to NeRF. It has been rapidly extended to dynamic scene reconstruction tasks through the integration of deformation fields [19] and further introduced for dynamic medical scene reconstruction. For example, EndoGaussian [11] models the deformation field with two lightweight modules and employs a multiresolution HexPlane [2] as the 4D structural encoder. Building on [9,8], Deform3DGS [22] discards the time-consuming MLP-based deformation fields in favor of basis functions to model Gaussian motion, further enhancing the efficiency of 3D reconstruction. Despite the improved rendering speed for real-time deformational medical applications, they still struggle to achieve accurate modeling of realistic colors and global deformations due to the following limitations: 1) Insufficient Color Expressiveness. In surgical scenes, tissues often exhibit similar appearances with subtle color variations. However, existing methods typically assign fixed color attributes to each Gaussian, leading to indistinguishable rendering results for similar tissues. 2) Locality of Gaussian Functions. Due to the computational inefficiency of MLP-based deformation fields, existing methods exploit linear combinations of basic functions with trainable parameters to model motion efficiently. However, the localized nature of Gaussian functions [22] makes it challenging to capture consistent global motion trends.

To this end, we propose ColorGS, a novel framework designed to enhance both color expressiveness and deformation modeling in 3DGS-based high-fidelity surgical scene reconstruction. Firstly, we introduce spatially adaptive color modeling (Colored Gaussians), which enhances the color expressiveness of individual Gaussians by integrating color information from dynamic anchors tailored for each Gaussian primitive. Furthermore, we propose a simple yet effective module, the Enhanced Deformation Model (EDM), which introduces a time-independent global deformation parameter to model consistent global motion and enhance the overall consistency and smoothness of the deformation. By jointly improving color expressiveness and deformation accuracy, ColorGS effectively enhances both the quality and reliability of surgical scene reconstruction. Extensive experiments on the EndoNeRF [18] and StereoMIS [4] datasets demonstrate that the proposed method achieves state-of-the-art reconstruction quality (PSNR: 39.85, SSIM: 97.25, LPIPS: 0.03).

## 2 Method

Pipeline. We propose ColorGS, a novel deformable surgical scene reconstruction paradigm that combines Colored Gaussians and EDM to achieve highquality and efficient 3D dynamic tissue reconstruction. As shown in Fig. 1, our method begins by using the corresponding camera parameters to obtain the point cloud of the first frame, which is then used to initialize the Gaussian primitives (Sec. 2.1). During the optimization, spatially adaptive anchor color aggregation is employed to enhance the color expressiveness of the Gaussian primitives (Sec. 2.2). Besides, a combination of time-aware basic functions and time-independent global deformation parameters is utilized to model the dynamic changes of the Gaussian primitivesâ properties (Sec. 2.3). Finally, the whole paradigm is trained using two loss functions that compare the ground-truth (GT) images with the rendered RGB/depth images (Sec. 2.4).

<!-- image-->  
Fig. 1. Pipeline of ColorGS, composed of (a) ColorGS Initialization, (b) Enhanced Deformational Modeling, and (c) Optimization using color and depth loss functions.

## 2.1 Preliminaries of 3D Gaussian Splatting

3DGS [6] introduces anisotropic 3D Gaussian parameters and a tile-based rasterizer to achieve high-quality and real-time 3D scene reconstruction. Each Gaus-

sian is parameterized by its center position $\mu ,$ covariance matrix $\Sigma ,$ opacity $^ { O , }$ and spherical harmonic (SH) coefficients. For an arbitrary coordinate $x ,$ the shape of the 3D Gaussian on x is described as:

$$
G ( x ) = e x p ( - \frac { 1 } { 2 } ( x - \mu ) ^ { T } \varSigma ^ { - 1 } ( x - \mu ) ) ,\tag{1}
$$

$$
\Sigma = R S S ^ { T } R ^ { T }\tag{2}
$$

where the covariance matrix $\varSigma$ is decomposed into a rotation matrix R and a diagonal scaling matrix $S _ { ; }$ allowing independent control over orientation and extent. All 3D Gaussians are projected onto the 2D image plane along the rays, enabling efficient rendering through fast Î±-blending. The positions $\mu ^ { 2 D }$ and the covariance matrices $\Sigma ^ { 2 D }$ of the projected 2D Gaussians can be analytically computed in the pixel coordinate system using the camera intrinsic and extrinsic parameters. To achieve view-independent rendering, the color $\hat { C } ( x )$ and the depth $\hat { D } ( x )$ of a center pixel x can be rendered by the function:

$$
\hat { C } ( x ) = \sum _ { i = 1 } ^ { n } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) , \hat { D } ( x ) = \sum _ { i = 1 } ^ { n } d _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } )\tag{3}
$$

where $\alpha _ { i }$ is given by evaluating a 2D covariance matrix $\Sigma ^ { 2 D }$ multiplied by the opacity $o _ { i } .$ , and $c _ { i }$ is the color computed from the SH coefficients of the i-th Gaussian, which is fixed for each Gaussian after the optimization process.

## 2.2 Colored Gaussian Primitives

In surgical scenes, variable lighting conditions and subtle color transitions often challenge the color expression capability of the original 3D Gaussians with fixed colors, leading to suboptimal reconstruction performance. We notice that adjacent tissues in surgical scenes typically have similar appearances, and identifying differences between these regions requires a careful comparison of neighboring pixels. This observation highlights the importance of incorporating spatial properties into the color modeling process for surgical scenes. Inspired by [5,7,21], we introduce a group of color anchors for each Gaussian to provide spatial-aware hints for optimizing its color representation, thereby enabling the capture of subtle color variations between similar tissues. These enhanced representations are referred to as Colored Gaussians.

Specifically, we introduce k dynamic anchors $A _ { i } = ( A _ { i } ^ { x } , A _ { i } ^ { y } )$ with color parameters $c _ { i }$ on the rendering plane for each Gaussian primitive, where $i \ =$ $0 , 1 , 2 , . . . , k - 1$ corresponds to the index of anchors. Then we use an exponential decay function to measure the contribution of each anchor to the color at the intersection point. When the ray intersects with the Gaussian primitive and generates the intersection point $\boldsymbol { p } = \left( u , v \right)$ on the rendering plane, the decay rate depends on the distance between p and the anchor coordinate $A _ { i }$ , which can be defined as:

$$
F _ { A _ { i } } ( p ) = e ^ { - \lambda _ { e } | | p - A _ { i } | | ^ { 2 } }\tag{4}
$$

where $\lambda _ { e }$ is used to control the rate of change. We set $\lambda _ { e } = 0 . 1$ and $k = 4$ by default. The added color produced by these anchors can be computed as:

$$
F _ { c } ( p ) = \sum _ { i = 0 } ^ { k - 1 } F _ { A _ { i } } ( p ) c _ { i }\tag{5}
$$

As a result, the color function for each Gaussian primitive is represented as:

$$
c ( p , d ) = S H ( d ) + F _ { c } ( p )\tag{6}
$$

where d represents the direction of the ray, $S H ( d )$ represents the spherical harmonics of direction d.

By introducing these anchors, each individual Gaussian can reflect different color variations at the intersection points, better fitting the details. Besides, since these anchors are defined on the rendering plane, the additional colors associated with the Colored Gaussians are inherently view-dependent, further enhancing spatially aware color modeling.

## 2.3 Enhanced Deformation Modeling

Modeling Gaussian motion through linear combinations of time-aware basis functions proves to be an efficient strategy [9,8,22], particularly for real-time surgical scene reconstruction. Among the available basis functions, Gaussian functions are preferred over the commonly used Fourier and polynomial basis functions. This preference arises from their ability to offer localized influence to preserve details without disrupting global motion. The Gaussian basis functions can be defined as:

$$
\tilde { b } ( t ; \theta _ { j } ^ { x } , \sigma _ { j } ^ { x } ) = e x p ( - \frac { ( t - \theta ) ^ { 2 } } { 2 \sigma ^ { 2 } } )\tag{7}
$$

where t represents time, and Î¸ and Ï correspond to the learnable center and variance, respectively.

However, due to the localized nature of Gaussian functions, modeling consistent long-term motion often requires a larger number of basis functions and involves more complex parameter optimization. Besides, when specific motion patterns arise, using linear combinations of Gaussian functions to accurately model them can be challenging. For instance, uniform motion requires a substantial number of Gaussian functions for proper fitting. To solve it, we propose EDM, which decouples motion into local dynamics represented by linear combinations of time-aware Gaussian functions and global motion trends modeled by time-independent global motion parameters. Specifically, the deformation parameters include the center position Âµ, the rotation matrix R, and the scaling matrix S. Taking the center position change of the Gaussian in the x-direction as an example, the position at any time t can be expressed as:

$$
\psi ^ { x } ( t , \Theta ^ { x } ) = \sum _ { j = 0 } ^ { B - 1 } \omega _ { j } ^ { x } \tilde { b } ( t ; \theta _ { j } ^ { x } , \sigma _ { j } ^ { x } ) + \delta _ { x }\tag{8}
$$

where B = 17 denotes the total number of basis functions. EDM decouples the global motion trend, which is challenging to represent with Gaussian functions, allowing for more flexible modeling of diverse motion patterns and enhancing the overall consistency and smoothness of the deformation.

## 2.4 Optimization

We optimize the whole paradigm by minimizing the discrepancy between the rendered outputs and the GT images. The whole training loss functions $\mathcal { L } _ { t o t a l }$ can be defined as:

$$
\mathcal { L } _ { t o t a l } = \Vert M \odot ( \hat { C } - C ) \Vert + \Vert M \odot ( \hat { D } - D ) \Vert\tag{9}
$$

where $\hat { C }$ and $\hat { D }$ denote the rendered RGB and depth images, $C$ and D represent the GT RGB and depth images, and M is the tissue mask.

## 3 Exeperiment

## 3.1 Experiment Setting

Dataset and Evaluation. Following [22], we evaluate the proposed method on two datasets: EndoNeRF [18] and StereoMIS [4]. EndoNeRF contains two cases of in-vivo prostatectomy data captured from stereo cameras at a single viewpoint, encompassing challenging scenes with non-rigid deformation and tool occlusion. The video clips in StereoMIS are captured from in-vivo porcine subjects and present additional challenges, including diverse anatomical structures and significant tissue deformations. All scenes of EndoNeRF and three clips from videos P2 and P3 in StereoMIS are used for performance evaluation. For each scene, the frames are divided into training and testing sets with a ratio of 7:1. To quantify the performance, we use PSNR, SSIM, and LPIPS as the metrics.

Implementation Details. For each scene, the video duration is normalized into [0, 1]. The training process spans 3000 iterations, starting with an initial learning rate of $1 . 6 \times 1 0 ^ { - 3 }$ . The densification on the Gaussian points number is frozen during the first 600 iterations for stable training. All the experiments are conducted with an NVIDIA RTX 3060 GPU.

## 3.2 Comparison with State-of-the-art Methods

Our proposed framework is compared with EndoNeRF and two recent 3DGSbased methods, i.e., EndoGaussian [11] and Deform3DGS [22]. EndoNeRF suffers from long training times and poor performance, severely limiting its practical use in surgery. EndoGaussian introduces Gaussian functions into dynamic surgical scene reconstruction and models dynamic Gaussians by decomposing the feature plane. Building on this, Deform3DGS further proposes the use of basis functions instead of traditional MLP networks for more efficient reconstruction. However, traditional 3D Gaussians struggle with limited color expressiveness, making it difficult to capture the similar appearance and subtle color variations of surgical tissues. Moreover, due to the localized nature of basis functions, capturing consistent global motion trends remains challenging. Our method overcomes these limitations, significantly improving both color accuracy and global motion consistency, resulting in higher-quality surgical scene reconstruction. As shown in Table 1 and Fig. 2, our method outperforms recent state-of-the-art surgical scene reconstruction results by a large margin. Specifically, on the EndoNeRF dataset, we achieve a remarkable PSNR improvement of 1.50 dB over the second-best method, Deform3DGS, while also improving SSIM and LPIPS.

<!-- image-->  
Fig. 2. Illustration of the rendered images of baselines and ours.

## 3.3 Quantitative Evaluation of Key Components

The Effect of Colored Gaussians. We conduct experiments using different 3D representations, including the original 3DGS and 2D Gaussian Splatting (2DGS) [3], to evaluate the color expression capabilities of our Colored Gaussian. As shown in Table 2, both 3DGS and 2DGS exhibit inferior performance compared to the proposed Colored Gaussian. Additionally, our method also outperforms SuperGaussian [21], which is built upon 2DGS, in terms of rendering quality. This demonstrates that the proposed method can effectively enhance the overall quality of the complex surgical scene reconstruction.

Table 1. Quantitative evaluation on endoscopic scene reconstruction. The best and suboptimal results are shown in bold and underlined respectively.
<table><tr><td>Dataset</td><td>Method</td><td>PSNRâ</td><td>SSIM(%)â</td><td>LPIPSâ</td></tr><tr><td rowspan="4">EndoNeRF</td><td>EndoNeRF</td><td>35.92</td><td>94.18</td><td>0.06</td></tr><tr><td>EndoGaussian</td><td>37.86</td><td>96.09</td><td>0.04</td></tr><tr><td>Deform3DGS</td><td>38.35</td><td>96.39</td><td>0.05</td></tr><tr><td>ours</td><td>39.85</td><td>97.25</td><td>0.03</td></tr><tr><td rowspan="4">StereoMIS</td><td>EndoNeRF</td><td>28.86</td><td>74.15</td><td>0.27</td></tr><tr><td>EndoGaussian</td><td>30.39</td><td>83.75</td><td>0.21</td></tr><tr><td>Deform3DGS</td><td>30.68</td><td>84.74</td><td>0.23</td></tr><tr><td>Ours</td><td>32.64</td><td>89.64</td><td>0.14</td></tr></table>

Table 2. Ablation study of the designed components on EndoNeRF.
<table><tr><td>Component</td><td>Method</td><td>PSNRâ</td><td>SSIM(%)â</td><td>LPIPSâ</td></tr><tr><td rowspan="3">Gaussian Splatting</td><td rowspan="3">2DGS 3DGS SuperGS</td><td>36.16</td><td>95.99</td><td>0.04</td></tr><tr><td>38.55</td><td>97.08</td><td>0.03</td></tr><tr><td>39.31</td><td>97.13</td><td>0.03</td></tr><tr><td rowspan="3">Gaussian tracking</td><td>Ours FPS</td><td>39.85 38.92</td><td>97.25 96.69</td><td>0.03 0.04</td></tr><tr><td>GS</td><td>39.54</td><td>97.13</td><td>0.03</td></tr><tr><td>Ours</td><td>39.85</td><td>97.25</td><td>0.03</td></tr></table>

The Effect of Enhanced Deformation Modeling. To investigate the effectiveness of EDM, we compare it with existing deformation modeling techniques on the EndoNeRF dataset. Specifically, we replace EDM with alternative methods, including the combination of Fourier and Polynomial series (FPS) and the Gaussian functions without global parameters (GS). As shown in Table 2, EDM achieves the best performance in representing deformations.

## 4 Conclusion

In this paper, we propose a high-quality dynamic modeling method for surgical scenes, called ColorGS. To enhance the expressive power of Gaussian representation, we develop Colored Gaussians, which introduce spatially varying colors to significantly improve the flexibility and precision of color representation for individual Gaussians. Additionally, to capture global deformation trends, we propose the Enhanced Deformation Model, which leverages basis functions and innovatively incorporates time-independent global motion parameters to ensure both global consistency and local adaptability. By seamlessly combining these two quality-enhancement strategies, we achieve high-quality surgical scene reconstruction with enhanced color details and deformation accuracy.

## References

1. Boedecker, C., Huettl, F., Saalfeld, P., Paschold, M., Kneist, W., Baumgart, J., Preim, B., Hansen, C., Lang, H., Huber, T.: Using virtual 3d-models in surgical planning: workflow of an immersive virtual reality application in liver surgery. Langenbeckâs archives of surgery 406, 911â915 (2021)

2. Cao, A., Johnson, J.: Hexplane: A fast representation for dynamic scenes. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 130â141 (2023)

3. Geiger, A., Gao, S., Chen, A., Yu, Z., Huang, B.: 2d gaussian splatting for geometrically accurate radiance fields (2024)

4. Hayoz, M., Hahne, C., Gallardo, M., Candinas, D., Kurmann, T., Allan, M., Sznitman, R.: Learning how to robustly estimate camera pose in endoscopic videos. International journal of computer assisted radiology and surgery 18(7), 1185â1192 (2023)

5. Kasymov, A., Czekaj, B., Mazur, M., Tabor, J., Spurek, P.: Neggs: Negative gaussian splatting. Information Sciences p. 121912 (2025)

6. Kerbl, B., Kopanas, G., LeimkÃ¼hler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42(4), 139â1 (2023)

7. Li, H., Liu, J., Sznaier, M., Camps, O.: 3d-hgs: 3d half-gaussian splatting. arXiv preprint arXiv:2406.02720 (2024)

8. Li, Z., Chen, Z., Li, Z., Xu, Y.: Spacetime gaussian feature splatting for realtime dynamic view synthesis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8508â8520 (2024)

9. Lin, Y., Dai, Z., Zhu, S., Yao, Y.: Gaussian-flow: 4d reconstruction with dynamic 3d gaussian particle. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21136â21145 (2024)

10. Liu, X., Stiber, M., Huang, J., Ishii, M., Hager, G.D., Taylor, R.H., Unberath, M.: Reconstructing sinus anatomy from endoscopic videoâtowards a radiation-free approach for quantitative longitudinal assessment. In: Medical Image Computing and Computer Assisted InterventionâMICCAI 2020: 23rd International Conference, Lima, Peru, October 4â8, 2020, Proceedings, Part III 23. pp. 3â13. Springer (2020)

11. Liu, Y., Li, C., Yang, C., Yuan, Y.: Endogaussian: Gaussian splatting for deformable surgical scene reconstruction. arXiv preprint arXiv:2401.12561 (2024)

12. Luo, H., Wang, C., Duan, X., Liu, H., Wang, P., Hu, Q., Jia, F.: Unsupervised learning of depth estimation from imperfect rectified stereo laparoscopic images. Computers in biology and medicine 140, 105109 (2022)

13. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65(1), 99â106 (2021)

14. Pelanis, E., Teatini, A., Eigl, B., Regensburger, A., Alzaga, A., Kumar, R.P., Rudolph, T., Aghayan, D.L., Riediger, C., KvarnstrÃ¶m, N., et al.: Evaluation of a novel navigation platform for laparoscopic liver surgery with organ deformation compensation using injected fiducials. Medical image analysis 69, 101946 (2021)

15. Rodby, K.A., Turin, S., Jacobs, R.J., Cruz, J.F., Hassid, V.J., Kolokythas, A., Antony, A.K.: Advances in oncologic head and neck reconstruction: systematic review and future considerations of virtual surgical planning and computer aided design/computer aided modeling. Journal of Plastic, Reconstructive & Aesthetic Surgery 67(9), 1171â1185 (2014)

16. Song, J., Wang, J., Zhao, L., Huang, S., Dissanayake, G.: Dynamic reconstruction of deformable soft-tissue with stereo scope in minimal invasive surgery. IEEE Robotics and Automation Letters 3(1), 155â162 (2017)

17. Tang, R., Ma, L.F., Rong, Z.X., Li, M.D., Zeng, J.P., Wang, X.D., Liao, H.E., Dong, J.H.: Augmented reality technology for preoperative planning and intraoperative navigation during hepatobiliary surgery: A review of current methods. Hepatobiliary & Pancreatic Diseases International 17(2), 101â112 (2018)

18. Wang, Y., Long, Y., Fan, S.H., Dou, Q.: Neural rendering for stereo 3d reconstruction of deformable tissues in robotic surgery. In: International conference on medical image computing and computer-assisted intervention. pp. 431â441. Springer (2022)

19. Wu, G., Yi, T., Fang, J., Xie, L., Zhang, X., Wei, W., Liu, W., Tian, Q., Wang, X.: 4d gaussian splatting for real-time dynamic scene rendering. In: CVPR. pp. 20310â20320. IEEE (2024)

20. Wu, T.Y., Meng, Q., Yang, L., Kumari, S., Pirouz, M.: Amassing the security: An enhanced authentication and key agreement protocol for remote surgery in healthcare environment. Comput. Model. Eng. Sci 134(1), 317â341 (2023)

21. Xu, R., Chen, W., Wang, J., Liu, Y., Wang, P., Gao, L., Xin, S., Komura, T., Li, X., Wang, W.: Supergaussians: Enhancing gaussian splatting using primitives with spatially varying colors. arXiv preprint arXiv:2411.18966 (2024)

22. Yang, S., Li, Q., Shen, D., Gong, B., Dou, Q., Jin, Y.: Deform3dgs: Flexible deformation for fast surgical scene reconstruction with gaussian splatting. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 132â142. Springer (2024)

23. Zha, R., Cheng, X., Li, H., Harandi, M., Ge, Z.: Endosurf: Neural surface reconstruction of deformable tissues with stereo endoscope videos. In: International conference on medical image computing and computer-assisted intervention. pp. 13â23. Springer (2023)

24. Zhou, H., Jagadeesan, J.: Real-time dense reconstruction of tissue surface from stereo optical video. IEEE transactions on medical imaging 39(2), 400â412 (2019)