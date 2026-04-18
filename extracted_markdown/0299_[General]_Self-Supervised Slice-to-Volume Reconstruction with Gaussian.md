# SELF-SUPERVISED SLICE-TO-VOLUME RECONSTRUCTION WITH GAUSSIAN REPRESENTATIONS FOR FETAL MRI

Yinsong Wang1 Thomas Fletcher1 Xinzhe Luo1 Aine Travers Dineen2 Rhodri Cusack2 Chen Qin1

1 Department of Electrical and Electronic Engineering and I-X, Imperial College London, London, UK 2 Trinity College Institute of Neuroscience, Trinity College Dublin

## ABSTRACT

Reconstructing 3D fetal MR volumes from motion-corrupted stacks of 2D slices is a crucial and challenging task. Conventional slice-to-volume reconstruction (SVR) methods are time-consuming and require multiple orthogonal stacks for reconstruction. While learning-based SVR approaches have significantly reduced the time required at the inference stage, they heavily rely on ground truth information for training, which is inaccessible in practice. To address these challenges, we propose GaussianSVR, a self-supervised framework for slice-to-volume reconstruction. GaussianSVR represents the target volume using 3D Gaussian representations to achieve high-fidelity reconstruction. It leverages a simulated forward slice acquisition model to enable self-supervised training, alleviating the need for ground-truth volumes. Furthermore, to enhance both accuracy and efficiency, we introduce a multiresolution training strategy that jointly optimizes Gaussian parameters and spatial transformations across different resolution levels. Experiments show that GaussianSVR outperforms the baseline methods on fetal MR volumetric reconstruction. Code will be available upon acceptance.

Index Termsâ Slice-to-volume reconstruction, Gaussian Representation, self-supervised learning

## 1. INTRODUCTION

High-resolution 3D fetal MRI is essential for advancing the understanding of fetal brain development [1]; however, it remains highly vulnerable to artifacts resulting from rapid and unpredictable fetal motion. To address this issue, twodimensional (2D) MRI techniques such as half-Fourier acquisition single-shot fast spin echo (SSFSE) [2] are used to acquire 2D slices within fractions of a second, effectively freezing in-plane motion. However, residual inter-slice motion and the use of thick slices to preserve signal-to-noise ratio (SNR) limit the ability of cross-sectional views to accurately capture the 3D brain structure, necessitating slice-to-volume reconstruction (SVR) to recover the underlying volume.

Conventional optimization-based slice-to-volume reconstruction (SVR) methods formulate the volumetric reconstruction as a joint registration and super-resolution problem. Rousseau et al. [3] and Gholipour et al [4] reconstruct a highresolution volume by iteratively estimating slice-to-volume transformations and voxel intensities under a slice acquisition model. Despite their effectiveness, the discretized voxel grid representation makes the complexity and memory footprint of SVR proportional to the number of voxels in the volume. Recently, several works have been proposed to use neural networks for SVR to alleviate the computational burden. Hou et al. [5] proposed a 3D CNN based on pre-trained VGG-16 convolutional layers, with a densely connected head to predict anchor points for individual slices. Xu et al. [6] introduced an iterative transformer to jointly estimate transformations and reconstruct the 3D volume. However, these methods require ground-truth transformations for training, which are inaccessible in practice. Subsequently, Xu et al. [7] utilized implicit neural representations (INR) to reconstruct the underlying volume, enabling a continuous and resolution-agnostic representation. However, the globally parameterized INR limits its capacity to adapt to local structural variability, resulting in suboptimal reconstruction performance.

To address the limitations, we propose to leverage 3D Gaussian representations to model the underlying 3D volumes. 3D Gaussian Splatting (3DGS) [8] emerged as a ground-breaking technique for novel-view synthesis due to its rapid rasterization and superior rendering quality in comparison to INR. It has also recently raised great interest in the medical imaging domain, including applications on CT reconstruction [9], surgical navigation [10], and surgical scene reconstruction [11]. However, the application of Gaussian representations to volumetric reconstruction and motion correction has not been explored in prior work. In this work, we present GaussianSVR, a self-supervised SVR framework based on Gaussian representations. Compared to the INRbased method, 3D Gaussian kernels offer spatially localized and independent primitives, enabling fine-grained adaptation to complex anatomical structure while preserving global consistency. Moreover, 3D Gaussian representations have an implicit regularization for the reconstructed volume due to the smooth nature of Gaussian kernels. To better handle Gaussian and motion estimation, we propose a multi-resolution training strategy that performs joint optimization of both parameters hierarchically across multiple resolution levels, as rigid motion can be more reliably estimated at coarser resolutions. Furthermore, we employ a simulated forward slice acquisition model to generate reconstructed stacks from the reconstructed volume and compute their discrepancy with the acquired slices, enabling self-supervised training. Our main contributions are summarized as follows:

<!-- image-->  
Fig. 1. Overview of the framework of the proposed GaussianSVR. Solid lines indicate forward propagation; dashed lines indicate backward propagation.

â¢ We are the first to propose the novel SVR framework based on 3D Gaussian representation.

â¢ We introduce a self-supervised multi-resolution training strategy for joint optimization of Gaussian and motion parameters with a simulated slice acquisition model.

â¢ Experiments show that GaussianSVR achieves superior reconstruction performance compared to the baseline methods.

## 2. METHODOLOGY

Given acquired stacks of 2D slices $\pmb { y } = [ y _ { 1 } , \dotsc , y _ { n } ]$ , the objective of slice-to-volume reconstruction (SVR) is to recover the underlying 3D volume xË. In the proposed method, xË is represented as a set of 3D Gaussian primitives. Specifically, our framework employs a simulated forward slice acquisition model to generate reconstructed stacks based on the estimated slice-wise transformations, thereby enabling self-supervised joint optimization of the Gaussian and transformation parameters through a multi-resolution training strategy. The overall framework is illustrated in Figure 1.

## 2.1. Volumetric Reconstruction by 3D Gaussian Representations

3D Gaussian Splatting (3DGS) [8] represents a 3D scene as a set of Gaussian primitives $G _ { j } \mid j = 1 , \ldots , J$ . Each Gaussian $G _ { j }$ is parameterized by its center $\mu _ { j } .$ , covariance $\Sigma _ { j } .$

opacity $\alpha _ { j }$ , and spherical harmonic coefficients $S H _ { j }$ for view-dependent appearance. Formally, a 3D scene can be expressed as $\boldsymbol { G } = \{ G _ { j } : \mu _ { j } , \Sigma _ { j } , \alpha _ { j } , S H _ { j } | 1 , . . . , J \}$ . To ensure the covariance matrix $\Sigma _ { j }$ remains positive semi-definite, it is decomposed into a rotation matrix $R _ { j }$ and a diagonal scaling matrix $S _ { j }$ as: $\Sigma _ { j } = R _ { j } S _ { j } ^ { 2 } R _ { j } ^ { T }$ . A 3D Gaussian primitive in continuous space is then defined as:

$$
G _ { j } ( { \bf x } ) = e ^ { - \frac { 1 } { 2 } ( { \bf x } - { \boldsymbol \mu } _ { j } ) ^ { T } \Sigma _ { j } ^ { - 1 } ( { \bf x } - { \boldsymbol \mu } _ { j } ) } ,\tag{1}
$$

where x denotes a spatial location in 3D space. Rendering a novel view of the scene involves projecting these 3D Gaussians onto the 2D image plane.

While the original 3DGS framework was developed for natural images, its appearance-related parameters, opacity $\alpha _ { j }$ and spherical harmonics $S H _ { j }$ , are inapplicable to medical imaging. Following Li et al. [9], we therefore remove these parameters and introduce an intensity coefficient $I _ { j }$ to represent the MRI intensity value at each Gaussian center. The 3D MRI volume is thus represented as $G = \{ G _ { j } : \mu _ { j } , \Sigma _ { j } , I _ { j } | 1 , . . . , J \}$ , and the contribution of each Gaussian to a spatial point x is formulated as

$$
G _ { j } ( { \bf x } | \mu _ { j } , \Sigma _ { j } , I _ { j } ) = I _ { j } e ^ { - \frac { 1 } { 2 } ( { \bf x } - \mu _ { j } ) ^ { T } \Sigma _ { j } ^ { - 1 } ( { \bf x } - \mu _ { j } ) } .\tag{2}
$$

The covariance matrix $\Sigma _ { j }$ is parameterized in the same way as in the original 3DGS [8] by a scaling matrix $S _ { j }$ and a rotation matrix $R _ { j }$ , such that $\Sigma _ { j } \ : \stackrel { \cdot } { = } \ : R _ { j } S _ { j } ^ { 2 } \bar { R } _ { j } ^ { T }$ to ensure the positive semi-definiteness of $\Sigma _ { j }$ and allow for independent optimization of these parameters.

To optimize the parameters of our 3D Gaussian representations and accurately estimate the MRI intensity values of the underlying 3D volume, we modify the standard rendering process used in original 3DGS [8]. Instead of employing the conventional splatting-based rendering, we compute the volumetric intensity at any spatial location x through a localized aggregation of contributions from neighboring Gaussians. To improve computational efficiency while maintaining reconstruction fidelity, we restrict the computation within a 99% $( \mu _ { j } \pm 3 \sigma _ { j } )$ confidence interval for each Gaussian $G _ { j }$ . Therefore, the MRI intensity value $V ( \mathbf { x } )$ for any give point x of the reconstructed 3D volume can be formulated as,

$$
V ( { \bf x } | \mu _ { j } , \Sigma _ { j } , I _ { j } ) = \sum _ { \substack { j : | | x - \mu _ { j } | | \le 3 \sigma _ { j } } } ^ { J } G _ { j } ( { \bf x } | \mu _ { j } , \Sigma _ { j } , I _ { j } ) .\tag{3}
$$

In this formulation, a 3D MR volume is represented as a set of Gaussian primitives, which allows spatially localized independent optimization for each primitive, enabling finegrained adaptation for complex anatomical structures.

## 2.2. Slice Acquisition Model

Given the underlying 3D volume x, the acquired stacks of 2D slices $\pmb { y } = [ y _ { 1 } , . . . , y _ { n } ] .$ , can be obtained by the forward slice

acquisition model, which is formulated as follows [6],

$$
y _ { i } = D B T _ { i } x ; i = 1 , . . . , n .\tag{4}
$$

Here $\mathbf { \nabla } _ { \mathbf { \boldsymbol { y } } _ { i } }$ represents the i th 2D slice, n is the total number of slices, $\mathbf { \delta } _ { \mathbf { \mathcal { T } } _ { i } }$ is the slice-wise transformation parameters of i th 2D slice, which describe the rotations and translations of i th plane within a canonical 3D atlas space. B represents the Point-Spread-Function (PSF) blurring matrix of the MRI signal acquisition process, and D is a down-sampling matrix. In this work, we modeled the PSF as an anisotropic 3D Gaussian distribution. Based on this, the reconstructed stack of slices $\pmb { \hat { y } } = [ \hat { y } _ { 1 } , . . . , \hat { y } _ { n } ]$ thus can be obtained using the reconstructed 3D volume xË through [6],

$$
\hat { y } _ { i } = D B \hat { T } _ { i } \hat { x } ; i = 1 , . . . , n ,\tag{5}
$$

where $\hat { \pmb { T } } _ { i }$ represents the optimized transformation parameters of i th slice from GaussianSVR.

## 2.3. Training

To improve both accuracy and efficiency, we propose a multiresolution training strategy to jointly optimize slice-wise transformations and the parameters of Gaussian representations across different spatial resolutions. The training process is formulated in a self-supervised manner without reliance on ground-truth supervision.

Low-Resolution Optimization. In the first stage, both the slice-wise transformations and Gaussian parameters are optimized at a coarse spatial resolution. Operating at low resolution stabilizes training and accelerates convergence, as the rigid slice-wise motion patterns are easier to capture when fine structural details are suppressed. This stage yields a robust initialization for the subsequent refinement phase.

High-Resolution Optimization. The second stage refines both Gaussian and transformation parameters obtained from the coarse-level optimization. By increasing the spatial resolution, GaussianSVR recovers fine-grained anatomical details while further improving alignment accuracy. This coarse-tofine strategy promotes a well-conditioned optimization landscape and mitigates convergence to local minima.

Self-supervised training via Slice Acquisition Model. An overview of the framework is illustrated in Figure 1. To enable self-supervised learning, the reconstructed 3D Gaussian volume is projected back into 2D slice space using the forward slice acquisition model described in Section 2.2. Given the estimated transformations $\hat { T }$ and the simulated pointspread function (PSF) in Section 2.2, the reconstructed stacks of slices $\hat { \pmb y } = [ \hat { y } _ { 1 } , \dots , \hat { y } _ { n } ]$ can then be obtained and compared with the acquired stacks $\pmb { y } = [ y _ { 1 } , \dots , y _ { n } ]$ to formulate the self-supervision. The Gaussian and transformation parameters are then jointly optimized by minimizing a reconstruction loss that combines an $\mathcal { L } _ { 1 }$ data fidelity term, a differentiable structural similarity (D-SSIM) term, and a total variation (TV) regularization term to promote spatial smoothness in the reconstructed volume. The overall loss function is defined as:

$$
\mathcal { L } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \bigl ( \| y _ { i } - \hat { y } _ { i } \| _ { 1 } + \lambda _ { 1 } D \mathbf { - S S I M } ( y _ { i } , \hat { y } _ { i } ) \bigr ) + \lambda _ { 2 } T V ( \hat { x } ) ,\tag{6}
$$

where $T V ( { \hat { x } } )$ denotes the total variation regularization applied to the reconstructed volume xË, and $\lambda _ { 1 }$ and $\lambda _ { 2 }$ denotes the respectively hyperparameter for D-SSIM and TV loss. This self-supervised formulation enables GaussianSVR to jointly optimize both transformation and Gaussian parameters directly from raw slice acquisitions, ensuring robust and high-fidelity volumetric reconstruction.

## 3. EXPERIMENTS

Dataset. We evaluate our proposed GaussianSVR on the Fetal Tissue Annotation Challenge (FeTA) dataset [12], which consists of T2-weighted (T2w) fetal brain MR images. We randomly selected 30 volumes as ground truths for evaluation. The volumes are registered to a fetal brain atlas [13], and resampled to the resolution of $0 . 8 \times 0 . 8 \times 0 . 8$ mm. We simulate 2D slices with a resolution of 1 mm Ã1 mm, slice thickness between 2.5 and 3.5 mm, and size of $1 2 8 \times 1 2 8 .$ For each subject, three image stacks comprising 15-30 slices were simulated along orthogonal orientations, following [6], with fetal brain motion trajectories generated following [14].

Evaluation Metrics. To evaluate the reconstruction accuracy with respect to the ground-truth volume, we assessed the reconstructed results using peak signal-to-noise ratio (PSNR), structural similarity index (SSIM) [15], and normalized root mean square error (NRMSE).

Baseline Methods. GaussianSVR is compared with three representative slice-to-volume reconstruction methods: the conventional NiftyMIC [16], the transformer-based learning approach SVoRT [6], and a recent optimization-based method using implicit neural representation, NeSVoR [7].

Implementation Details. The method was implemented in PyTorch and trained on an NVIDIA A6000 Ada GPU using the Adam optimizer [17]. The transformation parameters estimated by the pretrained SVoRT model [6] are employed as initialization for GaussianSVR. For multi-resolution training, the volume is reconstructed at a resolution downsampled by a factor of two during the low-resolution stage, and refined at full resolution during the high-resolution stage. For Gaussian parameters, the learning rate of the mean $\mu$ decayed from $2 \times 1 0 ^ { - 3 } \mathrm { ~ t o ~ 2 ~ } \times 1 0 ^ { - 6 }$ , while constant rates of 0.05, 0.005, and 0.001 were used for intensity I, scaling S, and rotation $R ,$ respectively. For transformation parameters, learning rates were set to $5 \times 1 0 ^ { - 4 }$ for translation and $5 \times 1 0 ^ { - 5 }$ for rotation.

Axial  
Coronal  
Sagittal  
<!-- image-->  
Fig. 2. Qualitative volumetric reconstruction results on a single subject from motion-corrupted scans on the FeTA dataset.

## 4. RESULTS AND DISCUSSION

Comparison studies. Table 1 reports the quantitative volumetric reconstruction performance of 3D fetal brain MRI on the FeTA dataset. We report the mean and standard deviation of the reconstruction results of the 30 test subjects. It can be observed that GaussianSVR achieves the highest reconstruction accuracy, achieving a 2.9% improvement in terms of PNSR compared to the second-best approach, NeSVoR. These results demonstrate that GaussianSVR substantially outperforms existing reconstruction approaches in both structural preservation and intensity consistency. The notably higher SSIM and lower NRMSE indicate that GaussianSVR more effectively maintains anatomical integrity and minimizes reconstruction artifacts. Figure 2 shows the qualitative reconstruction results of GaussianSVR compared with other baseline methods. It can be observed that GaussianSVR can reconstruct more fine-grained details. Compared to NeSVoR, GaussianSVR can reconstruct the volume with sharper details, demonstrating that Gaussian representations can achieve high-fidelity volumetric reconstruction.

Table 1. Quantitative Results on FeTA datasets. \* represents GaussianSVR significantly outperformed with p-value < 0.01 in a paired t-test. (Standard deviation in parentheses)
<table><tr><td>Methods</td><td>PSNR /dB â</td><td>SSIM â</td><td>NRMSE â</td></tr><tr><td>NiftyMIC</td><td>21.17*(1.95)</td><td>0.7653*(0.0559)</td><td>0.0989*(0.0234)</td></tr><tr><td>SVoRT</td><td>23.98*(2.65)</td><td>0.8209*(0.0618)</td><td>0.0905*(0.1227)</td></tr><tr><td>NeSVoR</td><td>25.58*(1.81)</td><td>0.8940*(0.0407)</td><td>0.0536(0.0105)</td></tr><tr><td>GaussianSVR (Ours)</td><td>28.19(3.02)</td><td>0.9281(0.0552)</td><td>0.0468(0.0219)</td></tr></table>

Table 2. Ablation study on each module of GaussianSVR. \* represents GaussianSVR significantly outperformed with pvalue < 0.01 in a paired t-test. (Standard deviation in parentheses)
<table><tr><td>Methods</td><td>PSNR / dB â</td><td>SSIM â</td></tr><tr><td>w/o low resolution</td><td>27.08*(3.89)</td><td>0.9134*(0.0547)</td></tr><tr><td>w/o transformation optimization</td><td>22.86*(2.38)</td><td>0.8148*(0.0752)</td></tr><tr><td>GaussianSVR (Ours)</td><td>28.19(3.02)</td><td>0.9281(0.0552)</td></tr></table>

Ablation Studies. We conducted ablation studies to assess the effectiveness of the proposed multi-resolution and joint optimization strategy. The results are summarized in Table 2. Removing multi-resolution training degrades model performance, likely because slice-wise transformations are more stable and converge more effectively at lower resolutions. Omitting transformation optimization causes a substantial performance drop, indicating that jointly optimizing transformations helps the model escape local minima and achieve improved global convergence.

## 5. CONCLUSION

In this work, we propose GaussianSVR, a self-supervised slice-to-volume reconstruction (SVR) framework based on 3D Gaussian representations. GaussianSVR employs 3D Gaussian kernels to model the volumetric structure, enabling high-fidelity reconstruction through spatially localized and independent primitives that facilitate fine-grained detail reconstruction. A multi-resolution training strategy is proposed to jointly optimize Gaussian and transformation parameters, leading to more accurate motion estimation and reconstruction. Furthermore, a simulated forward slice acquisition model allows self-supervised training by generating reconstructed stacks of slices and comparing them with the acquired stacks of slices. Experimental results demonstrate that GaussianSVR outperforms baseline methods in reconstruction quality. Future work will explore the resolution-agnostic capability of GaussianSVR and its potential for volumetric reconstruction from a single stack of slices.

## 6. ACKNOWLEDGMENTS

This work was supported by the Engineering and Physical Sciences Research Council [grant number EP/Y002016/1] and by Research Ireland under FreezeMotion project [grant number 22/FFP-A/11050]. X. Luo was supported by the Engineering and Physical Sciences Research Council [grant number EP/X039277/1].

## 7. COMPLIANCE WITH ETHICAL STANDARDS

This research study utilized publicly available human subject data from the Fetal Tissue Annotation Challenge (syn25649159), for which ethical approval was obtained by the original data collectors as reported in the associated publications.

## 8. REFERENCES

[1] Oualid Benkarim, Gemma Piella, Islem Rekik, Nadine Hahner, Elisenda Eixarch, Dinggang Shen, Gang Li, Miguel Angel GonzÃ¡lez Ballester, and Gerard Sanroma, âA novel approach to multiple anatomical shape analysis: application to fetal ventriculomegaly,â Medical image analysis, vol. 64, pp. 101750, 2020.

[2] Sahar N Saleem, âFetal mri: An approach to practice: A review,â Journal of advanced research, vol. 5, no. 5, pp. 507â523, 2014.

[3] Francois Rousseau, Orit A Glenn, Bistra Iordanova, Claudia Rodriguez-Carranza, Daniel B Vigneron, James A Barkovich, and Colin Studholme, âRegistration-based approach for reconstruction of high-resolution in utero fetal mr brain images,â Academic radiology, vol. 13, no. 9, 2006.

[4] Ali Gholipour, Judy A Estroff, and Simon K Warfield, âRobust super-resolution volume reconstruction from slice acquisitions: application to fetal brain mri,â IEEE transactions on medical imaging, vol. 29, no. 10, pp. 1739â1758, 2010.

[5] Benjamin Hou, Bishesh Khanal, Amir Alansary, Steven McDonagh, Alice Davidson, Mary Rutherford, Jo V Hajnal, Daniel Rueckert, Ben Glocker, and Bernhard Kainz, â3-d reconstruction in canonical co-ordinate space from arbitrarily oriented 2-d images,â IEEE transactions on medical imaging, vol. 37, no. 8, 2018.

[6] Junshen Xu, Daniel Moyer, P Ellen Grant, Polina Golland, Juan Eugenio Iglesias, and Elfar Adalsteinsson, âSvort: Iterative transformer for slice-to-volume registration in fetal brain mri,â in MICCAI, 2022.

[7] Junshen Xu, Daniel Moyer, Borjan Gagoski, Juan Eugenio Iglesias, P. Ellen Grant, Polina Golland, and Elfar

Adalsteinsson, âNesvor: Implicit neural representation for slice-to-volume reconstruction in mri,â IEEE Transactions on Medical Imaging, vol. 42, no. 6, pp. 1707â 1719, 2023.

[8] Bernhard Kerbl, Georgios Kopanas, Thomas LeimkÃ¼hler, and George Drettakis, â3d gaussian splatting for real-time radiance field rendering.,â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[9] Yingtai Li, Xueming Fu, Han Li, Shang Zhao, Ruiyang Jin, and S Kevin Zhou, â3dgr-ct: Sparse-view ct reconstruction with a 3d gaussian representation,â Medical Image Analysis, p. 103585, 2025.

[10] Maximilian Fehrentz, Alexander Winkler, Thomas Heiliger, Nazim Haouchine, Christian Heiliger, and Nassir Navab, âBridgesplat: Bidirectionally coupled ct and non-rigid gaussian splatting for deformable intraoperative surgical navigation,â in MICCAI, 2025.

[11] Thatphum Paonim, Chayapon Sasnarukkit, Natawut Nupairoj, and Peerapon Vateekul, âEndoplanar: Deformable planar-based gaussian splatting for surgical scene reconstruction,â in MICCAI, 2025.

[12] Kelly Payette, Priscille de Dumast, Hamza Kebiri, Ivan Ezhov, Johannes C Paetzold, Suprosanna Shit, Asim Iqbal, Romesa Khan, Raimund Kottke, Patrice Grehten, et al., âAn automatic multi-tissue human fetal brain segmentation benchmark using the fetal tissue annotation dataset,â Scientific data, vol. 8, no. 1, pp. 167, 2021.

[13] Ali Gholipour, Clemente Velasco-Annis, Caitlin K. Rollins, Lana Vasung, Abdelhakim Ouaalam, Cynthia Ortinau, Alireza Akhondi-Asl, Sean Clancy, Edward Yang, Judy Estroff, and Simon K. Warfield, âIMAGINE Fetal T2-weighted MRI Atlas,â 2023.

[14] Junshen Xu, Esra Abaci Turk, P Ellen Grant, Polina Golland, and Elfar Adalsteinsson, âStress: Superresolution for dynamic fetal mri using self-supervised learning,â in MICCAI, 2021.

[15] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli, âImage quality assessment: from error visibility to structural similarity,â IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600â612, 2004.

[16] Michael Ebner, Guotai Wang, Wenqi Li, Michael Aertsen, Premal A Patel, Rosalind Aughwane, Andrew Melbourne, Tom Doel, Steven Dymarkowski, et al., âAn automated framework for localization, segmentation and super-resolution reconstruction of fetal brain mri,â NeuroImage, vol. 206, pp. 116324, 2020.

[17] Kingma DP Ba J Adam et al., âA method for stochastic optimization,â arXiv preprint arXiv:1412.6980, vol. 1412, no. 6, 2014.