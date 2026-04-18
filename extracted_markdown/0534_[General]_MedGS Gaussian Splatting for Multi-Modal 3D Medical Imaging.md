# MedGS: Gaussian Splatting for Multi-Modal 3D Medical Imaging

Kacper Marzol Jagiellonian University

Ignacy Kolton Jagiellonian University

Joanna Kaleta Warsaw University of Technology Sano Centre for Computational Medicine

Weronika Smolak-Dyzewska Ë Jagiellonian University

Marcin Mazur Jagiellonian University

PrzemysÅaw Spurek Jagiellonian University IDEAS Research Institute

przemyslaw.spurek@uj.edu.pl

INTERPOLATION

<!-- image-->  
DEFORMATION

<!-- image-->

FOLDED GAUSSIANS 3D RECONSTRUCTION

<!-- image-->

<!-- image-->

MESH EXTRACTION

<!-- image-->

Figure 1. Overview of the MedGS. Input consists of 2D transverse medical images paired with corresponding 3D freehand poses which are used to reconstruct a scene with Folded Gaussians. The grayscale frames can be then used for interpolation and deformation. Binary masks can be utilized for mesh extraction. The VeGaS-based framework models temporal transitions, with In-Between Frame Regularization (IBFR) improving interpolation quality. Final outputs include high-fidelity interpolated frames and smooth mesh reconstructions.

## Abstract

Multi-modal three-dimensional (3D) medical imaging data, derived from ultrasound, magnetic resonance imaging (MRI), and potentially computed tomography (CT), provide a widely adopted approach for non-invasive anatomical visualization. Accurate modeling, registration, and visualization in this setting depend on surface reconstruction and frame-to-frame interpolation. Traditional methods often face limitations due to image noise and incomplete in-

formation between frames. To address these challenges, we present MedGS, a semi-supervised neural implicit surface reconstruction framework that employs a Gaussian Splatting (GS)-based interpolation mechanism. In this framework, medical imaging data are represented as consecutive two-dimensional (2D) frames embedded in 3D space and modeled using Gaussian-based distributions. This representation enables robust frame interpolation and highfidelity surface reconstruction across imaging modalities. As a result, MedGS offers more efficient training than traditional neural implicit methods. Its explicit GS-based representation enhances noise robustness, allows flexible editing, and supports precise modeling of complex anatomical structures with fewer artifacts. These features make MedGS highly suitable for scalable and practical applications in medical imaging. The source code and additional visualizations are available at https://github.com/gmum/ MedGS.

## 1. Introduction

Multi-modal three-dimensional (3D) medical imaging, encompassing ultrasound [17], magnetic resonance imaging (MRI) [5], and computed tomography (CT) [14], has become an indispensable tool for non-invasive anatomical visualization. Each modality has its own strengths, with ultrasound being particularly valued in clinical practice due to its cost-effectiveness, absence of ionizing radiation, and real-time imaging capability.

Within 3D medical imaging, surface reconstruction and frame-to-frame interpolation are key processes that enable accurate analysis and visualization. Conventional approaches, such as contour filtering or marching cubes [29], reconstruct surfaces by converting volumetric voxel data into polygonal or triangulated meshes, typically guided by segmentation boundaries [27]. However, these methods often suffer from connectivity artifacts, holes, and rough surfaces caused by noise or incomplete data. Post-processing steps like smoothing and interpolation are usually required [47], yet the final quality remains heavily dependent on the resolution and fidelity of the underlying voxel data [30].

In recent years, deep learning (DL) has emerged as a powerful alternative, offering superior accuracy, faster inference, and robustness to heterogeneous inputs [9]. Among DL-based strategies, implicit neural representations (INRs) have gained considerable attention [19, 35]. INRs parameterize 3D structures with neural implicit functions, commonly trained to represent signed distance functions (SDFs) [31, 32]. Such methods have already been applied in medical imaging for a wide range of tasks, including slice-tovolume MRI reconstruction [45], limited-view CT reconstruction [37], and freehand 3D ultrasound reconstruction [8, 11, 44, 46]. Beyond volumetric data, INRs have demonstrated strong performance for surface-specific applications, such as reconstructing the abdominal aorta from robotic ultrasound [38], cortical surfaces from MRI [6], and the left ventricle from sparse cardiac MRI [33]. Other studies extended INRs to capture biological dynamics, for example, modeling shape variations of living cells from microscopy [43], or representing organ structures from sparse CT and MRI scans using occupancy functions [1].

While INRs achieve high-quality reconstructions, they come with important limitations: training is computationally demanding, fine-tuning requires significant resources, and the resulting implicit models are challenging to interpret or manually edit. Recently, Gaussian Splatting (GS) [21] was introduced as an efficient representation, using sets of Gaussian primitives to model 3D structures. This formulation provides several advantages over INRs: training can be performed more efficiently, reconstructions remain editable, and the representation is inherently interpretable.

The ability to edit medical images is crucial for applications like ultrasound registration in computer-aided surgery. This is because deformable registration is often required to account for tissue deformation that occurs between pre-operative and interventional images [10, 41]. While tools like SynthMorph [16] have addressed this for brain MRI, the challenge of recovering morphologically accurate anatomical images from deformed ultrasound scans remains critical for precise and consistent diagnosis, particularly in computer-assisted fields [18]. An editable representation is therefore essential for all these tasks.

In this work, we introduce MedGS, the first GS-based framework for multi-modal 3D medical imaging. Our approach treats volumetric medical data as sequences of 2D cross-sections embedded in 3D space. Transitions between adjacent slices are modeled using a 3D GS representation, enabling robust interpolation and surface reconstruction. To enhance modeling fidelity, we build upon VeGaS [36], an extension of GS that incorporates Folded-Gaussian distributions. This adaptation allows MedGS to capture more complex and nonlinear structures while preserving the advantages of explicit Gaussian representations, including editability and efficient training. Additionally, MiraGe [40] provides a frame-level Gaussian Splatting backbone, allowing each 2D image to be represented consistently within the 3D context.

The contributions of this work are as follows:

â¢ We propose MedGS, the first Gaussian Splatting-based framework specifically designed for multi-modal 3D medical imaging. This novel approach leverages an efficient GS-based representation to model complex anatomical structures with high fidelity.

â¢ MedGS supports robust frame interpolation and accurate surface or mesh reconstruction across diverse imaging modalities, enabling improved visualization and analysis from sparse or noisy data.

â¢ Our framework preserves the inherent editability of GSbased models, allowing direct and intuitive manipulation of 3D medical data. This flexibility facilitates postprocessing, error correction, and customization in clinical and research applications.

## 2. Related Works

Traditional methods for surface reconstruction and frameto-frame interpolation in multi-modal 3D medical imaging data are generally categorized as either direct extraction or indirect segmentation approaches. Early direct methods, such as those by Zhang et al. [47, 48], utilized ISO-Surface extraction from freehand ultrasound data to generate surfaces directly. Similarly, Kerr et al. [22] applied columnwise thresholding on synthetic aperture images to produce point clouds, which were subsequently converted into surface meshes via wrapping algorithms. Indirect approaches typically involve first segmenting contours of the region of interest (ROI); for instance, Nguyen et al. [30] used Bezierspline interpolation on segmented ultrasound contours to construct high-quality triangular meshes. However, these classical methods often struggle with noise and incomplete data, resulting in rough or disconnected reconstructions.

<!-- image-->  
Figure 2. Qualitative comparison of mesh reconstruction methods. Each subfigure highlights the preservation of topology, edge smoothness, and overall surface geometry. FUNSR and Poisson methods fail to produce smooth meshes, likely due to the limited number of training frames.

In recent years, deep learning has become a powerful alternative across various modalities, including ultrasound, MRI, and CT. Wang et al. [42] extended graph convolutional networks (GCNs) to learn mesh deformations from CT images, while Nakao et al. [28] combined convolutional neural networks (CNNs) and GCNs for 3D liver surface reconstruction from single 2D radiographs. Other works have developed end-to-end frameworks to convert volumetric data directly into meshes, facilitating joint segmentation and surface reconstruction [13, 50]. In brain MRI, deformable template-based geometric networks have gained traction for cortical surface modeling [3, 25, 34], and weakly supervised autoencoders have been proposed for 4D shape mesh generation from 2D echocardiographic videos [23].

Implicit neural representations (INRs) represent a recent frontier in volume and surface reconstruction. INRs have been applied to MRI slice-to-volume reconstruction [45], limited-view CT [37], and freehand 3D ultrasound [8, 11, 44, 46]. For surface reconstruction, end-to-end signed distance function (SDF) learning has become predominant [2, 49], evolving from early methods such as DeepSDF [31] that required ground truth distances to recent approaches learning directly from point clouds [24]. INRbased techniques have been further adapted to reconstruct cortical brain surfaces [6], high-resolution left ventricular shapes from sparse cardiac MRI views [33], and to encode temporal dynamics for modeling living cell morphologies from microscopy [43]. While many focus on implicit occupancy functions [1, 26], the FUNSR method [4] simultaneously learns a geometric surface and signed distance field.

In contrast to neural network-based methods, MedGS leverages a Gaussian Splatting-based representation, enabling efficient and explicit modeling, particularly suitable for multi-modal 3D medical imaging.

<!-- image-->  
Figure 3. Examples of structural edits performed with MedGS on a brain MRI. By directly modifying the Gaussian components, realistic deformations of the object are achieved.

## 3. Preliminaries

Our method, MedGS, builds upon two foundational models: Gaussian Splatting (GS) [21] and Video Gaussian Splatting (VeGaS) [36]. We provide a brief overview of both approaches in the following paragraphs.

Gaussian Splatting Gaussian Splatting (GS) [21] is a powerful explicit 3D scene representation method that models a scene as a collection of Gaussian primitives. Each Gaussian is parameterized by its mean (position), an anisotropic covariance matrix that captures its shape and orientation, an opacity value, and view-dependent color represented using spherical harmonics. Formally, a GS scene is described as

$$
\mathcal { G } _ { \mathrm { G S } } = \{ ( \mathcal { N } ( m _ { i } , \pmb { \Sigma } _ { i } ) , \rho _ { i } , \mathbf { c } _ { i } ) \} _ { i = 1 } ^ { n } ,\tag{1}
$$

where $\mathbf { m } _ { i }$ and $\Sigma _ { i }$ denote the mean and covariance matrix of the i-th Gaussian, $\rho _ { i }$ is opacity, and $\mathbf { c } _ { i }$ means the spherical harmonics color coefficients. GS employs a rendering pipeline that projects these 3D Gaussian components onto the image plane, allowing efficient and differentiable rendering suitable for iterative optimization by comparing synthetic views with training images. This approach enables fast reconstruction of detailed and realistic 3D scenes with explicit and interpretable representations, facilitating editing and compositional operations. Though initially developed for static general 3D scenes, recent works [12, 15, 39] have extended GS to applications such as mesh reconstruction. However, adaptation to more specialized domains like medical imaging, characterized by sparse sampling and noisy or incomplete data, remains challenging and motivates further refinement [21].

Video Gaussian Splatting (VeGaS) Video Gaussian Splatting (VeGaS) [36] builds upon GS to model dynamic sequences such as video by introducing Folded-Gaussian distributions, which generalize classical Gaussians to better capture temporal and nonlinear spatial dynamics in video data. In VeGaS, a video is represented as a sequence of frames positioned as parallel planes in 3D space. Each scene element is parameterized as a 3D Folded-Gaussian that conditions its spatial distribution on frame timestamps. This temporal conditioning is realized through learnable functions applied to the mean and covariance of each Gaussian, resulting in time-dependent transformations that adapt Gaussian shape and position flexibly:

<!-- image-->  
Figure 4. Comparison between the reconstructed mesh from the annotated Prostate Ultrasound dataset and its modified counterpart. The modification stretches the mesh at the beginning and end along the Y-axis, while the central region is elongated along the X-axis. All views are presented from a specific axis projection.

$$
\begin{array} { r l } & { \mathcal { N } ( \mathbf { m } _ { s \mid t } , \Sigma _ { s \mid t } , a , f ) ( \mathbf { s } \mid t ) } \\ & { \qquad = \mathcal { N } ( \mathrm { m } _ { \mathrm { s } } + f ( m _ { t } - t ) , a ( t ) \Sigma _ { \mathrm { s } } ) ( \mathrm { s } \mid t ) , } \end{array}\tag{2}
$$

where $a ( t )$ and f(t) encode temporal rescaling and nonlinear (typically polynomial) shifts. The full Folded-Gaussian probability density then factors as

$$
\begin{array} { r l } & { \mathcal { F N } ( \mathbf { m } , \Sigma , a , f ) ( \mathbf { x } ) } \\ & { \qquad = \mathcal { N } ( \mathbf { m } _ { s \mid t } , \Sigma _ { s \mid t } , a , f ) ( \mathbf { s } \mid t ) \cdot \mathcal { N } ( m _ { t } , \sigma _ { t } ^ { 2 } ) ( t ) , } \end{array}\tag{3}
$$

with $\mathbf { x } ~ = ~ ( \mathbf { s } | t , t )$ representing the joint space-time variable. This scheme allows VeGaS to efficiently and accurately model both persistent features (via broadly spanning Gaussians) and ephemeral events (via tightly localized

Gaussians) across video frames. The explicit, editable representation permits superior frame interpolation, dynamic scene reconstruction, and interactive video editing capabilities. VeGaS thus overcomes many limitations of static GS and implicit neural representations for dynamic data, providing a versatile and practical model for spatiotemporal medical data, where frames occur at fixed intervals and noise or missing data are common [36].

## 4. MedGS

This section presents our MedGS model, which integrates VeGaS within a Gaussian Splatting-based framework. As illustrated in Figure 1, training employs a sequence of 2D transverse images paired with corresponding 3D freehand poses. Original photos are used for the interpolation task, while binary masks guide mesh reconstruction.

Interpolation Task The input consists of medical data represented as a sequence of grayscale frames $[ I _ { t _ { 1 } } , \ldots , I _ { t _ { n } } ]$ , indexed by their occurrence times normalized to the unit interval [0, 1]. Formally, MedGS is defined as a collection of 3D Folded-Gaussians:

$$
\mathcal { G } _ { \mathrm { M e d G S } } = \{ ( \mathcal { F N } ( \mathbf { m } _ { i } , \boldsymbol { \Sigma } _ { i } , a _ { i } , f _ { i } ) , \rho _ { i } , c _ { i } ) \} _ { i = 1 } ^ { n } ,\tag{4}
$$

where the colors $c _ { i }$ are grayscale.

Medical data are typically noisy, which can adversely affect GS-based methods. Although image blurring can mitigate noise, it also compromises rendering quality. To address this, MedGS introduces an In-Between Frame Regularization (IBFR) technique that both reduces noise and regularizes Folded-Gaussian components. For each consecutive frame pair $[ I _ { t } , I _ { t + 1 } ]$ , an interpolated frame is generated as follows:

$$
I _ { t _ { \alpha } } = \alpha \cdot I _ { t } + ( 1 - \alpha ) \cdot I _ { t + 1 } ,\tag{5}
$$

where Î± is uniformly sampled between 0.2 and 0.8 (this range was empirically chosen based on preliminary experiments). These interpolated frames are incorporated into the training set, serving as a regularizer and introducing an additional loss term, $\mathcal { L } _ { \mathrm { i n t e r p } } ^ { \ast } .$ Folded-Gaussian components are required to approximate both the original and the interpolated frames, which reduces noise and prevents the model from bending Folded-Gaussians between slices in an unconstrained way. This regularization encourages appropriate fitting of Gaussian tails, enabling accurate interpolation on test data. IBFR thus provides a simple yet highly effective mechanism for enhancing interpolation performance in GS-based models.

To further enforce temporal consistency, we constrain the time-spread parameter $\sigma _ { t }$ of each Folded-Gaussian (see Eq. (3)). Specifically, we penalize $\sigma _ { t }$ values that exceed 1 or fall below $\textstyle { \frac { 2 } { N } }$ , where N denotes the number of frames in the training set, resulting in the loss term $\mathcal { L } _ { \sigma }$ . This approach encourages each Folded-Gaussian to span multiple frames rather than collapsing to a single slice, while also preventing overly diffuse temporal components.

MedGS employs a VeGaS-based approach without adaptive frame fitting, necessitating fixed frame intervals for IBFR. Both original and interpolated images are utilized to train Folded-Gaussians. The densification and pruning procedure from the original Gaussian Splatting [21] is applied, wherein new components are added and redundant ones are removed during training.

Mesh Reconstruction Task Interpolation between frames enhances reconstruction, visualization, and partial object inpainting. However, the primary objective is mesh reconstruction for annotated objects. In this context, binary masks obtained via manual annotation or segmentation algorithms are used as input to generate a mesh that accurately represents the masked regions.

For mesh reconstruction, MedGS trains Folded-Gaussian components using binary images as input. Although similar training procedures as those used for interpolation may be applied, In-Between Frame Regularization is unnecessary because these masks are generally noise-free. After training, we interpolate the binary masks to generate a dense sequence of images. Applying the classical marching cubes algorithm [29] to this sequence produces a highquality mesh reconstruction.

Loss Functions Training objectives combine pixel fidelity, perceptual similarity, interpolation consistency, and temporal regularization.

For the interpolation task, the total loss is defined as follows:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { i n t e r p } } = \lambda _ { 1 } \mathcal { L } _ { \mathrm { L 1 } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { S S I M } } + \lambda _ { 3 } \mathcal { L } _ { \mathrm { i n t e r p } } ^ { * } + \lambda _ { 4 } \mathcal { L } _ { \sigma } , } \end{array}\tag{6}
$$

where ${ \mathcal { L } } _ { \mathrm { L 1 } }$ and $\mathcal { L } _ { \mathrm { i n t e r p } } ^ { \ast }$ denote the mean absolute errors between the reconstructed original frames and the interpolated frames with their respective predictions. The term LSSIM encourages structural similarity in the reconstructed original frames, while $\mathcal { L } _ { \sigma }$ regularizes the temporal spread of the Folded-Gaussian components:

$$
\begin{array} { r } { \mathcal { L } _ { \sigma } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left[ \operatorname* { m a x } \left( \frac { 2 } { N } - \sigma _ { t } , 0 \right) + \operatorname* { m a x } \left( \sigma _ { t } - 1 , 0 \right) \right] . } \end{array}\tag{7}
$$

For mesh reconstruction, noise is less significant, so the objective simplifies to

$$
\mathcal { L } _ { \mathrm { m e s h } } = \lambda _ { 1 } \mathcal { L } _ { \mathrm { L 1 } } + \lambda _ { 4 } \mathcal { L } _ { \sigma } .\tag{8}
$$

MedGS Editing One of the key advantages of MedGS over INR-based models is its editing capacity. In both interpolation and mesh reconstruction tasks, our model is composed of 3D Folded-Gaussians. Conditioning the Folded-Gaussian distribution on the time variable produces 2D

<!-- image-->  
Figure 5. Qualitative results of mesh reconstruction on the prostate ultrasound dataset from the MICCAI 2023 MRI-to-US Registration for Prostate (Âµ-RegPro) Challenge. MedGS produces smooth shapes that most closely match the ground truth (GT) mesh.

<!-- image-->  
Figure 6. Qualitative results of frame interpolation on a brain MRI volume. Our method (MedGS) produces a sharper reconstruction.

Gaussian components defined on planes, specifically yielding the MiraGe representation introduced in [40]. Each 2D Gaussian is concretely represented by three points corresponding to the Gaussian center and the two eigenvectors of its covariance matrix. These points can be manually adjusted to generate realistic edits of the resulting image. We demonstrate this approach with grayscale Gaussians in the interpolation task (see Figure 3) and with black-and-white Gaussians in the mesh reconstruction task (see Figures 3 and 4).

## 5. Experiments

We evaluate MedGS on two fundamental tasks in multimodal 3D medical imaging: frame interpolation and mesh reconstruction. Our experiments are designed to demonstrate the methodâs accuracy and robustness, particularly when working with sparse or incomplete data.

## 5.1. Datasets

We conduct experiments using two imaging modalities: Ultrasound and Magnetic Resonance Imaging (MRI).

Ultrasound We utilize the publicly available Prostate Ultrasound dataset from the MICCAI 2023 MRI-to-US Registration for Prostate (Âµ-RegPro) Challenge, which contains transrectal ultrasound (TRUS) volumes from 108 patients. Of the 73 publicly released volumes, 65 are originally assigned for training and 8 for validation. Each volume includes clearly annotated anatomical landmarks such as the prostate gland, visible lesions, and zonal structures. In our experiments, we incorporate the validation cases into training and report performance metrics per patient, as well as overall averages.

Magnetic Resonance Imaging (MRI) We employ two MRI datasets for interpolation and mesh reconstruction: TotalSegmentator MRI and the Skull-Stripped T1-Weighted MRI of Huffaz and Non-Huffaz Brains. The TotalSegmentator benchmark [7] provides sequence-independent, wholebody MRI scans with consistent multi-organ manual annotations. We use abdominal and pelvic subsets containing high-quality volumetric masks of organs such as the kidney, heart, and lung, making this dataset well-suited for mesh reconstruction.

The Skull-Stripped T1-Weighted MRI of Huffaz and Non-Huffaz Brains dataset, recently released on Open-Neuro, includes high-resolution T1-weighted brain scans with skull-stripping applied to remove non-brain tissue. The resulting clean volumetric data is particularly suitable for evaluating slice interpolation accuracy.

Together, these datasets allow us to assess MedGS under realistic MRI contrast and noise conditions for both multiorgan surface reconstruction and cross-slice interpolation.

## 5.2. Baselines

We compare MedGS against representative baselines for both frame interpolation and mesh reconstruction.

For frame interpolation, we consider (i) Linear Interpolation, a straightforward voxel-wise method between adjacent slices commonly used in medical imaging, and (ii) Optical Flow, which estimates dense flow fields between neighboring slices to warp intermediate frames.

Linear Interpolation  
Optical Flow  
<!-- image-->  
MedGs (Ours)  
Figure 7. Qualitative results of frame interpolation on an ankle MRI volume. Our method (MedGS) produces a sharper reconstruction. This improvement is clearly evident in the difference map between the reconstructed and original images.

Table 1. MRI interpolation results using every 2nd, 3rd, and 5th frame for training. Metrics reported include PSNR, Dice Coefficient, IoU, and SSIM, where higher values indicate better performance. Boldface highlights the best result in each column.
<table><tr><td colspan="3">PSNR â Dice Coeff. â IoU â SSIM â</td></tr><tr><td></td><td>Every 2nd frame</td><td></td></tr><tr><td>Linear Optical Flow MedGS</td><td> $3 2 . 7 0 \pm 1 . 9 0$   $0 . 5 7 \pm 0 . 2 7$   $2 9 . 0 7 \pm 2 . 2 7$   $0 . 5 0 \pm 0 . 2 5$   $3 3 . 5 2 \pm 1 . 8 7$   ${ \bf 0 . 6 7 \pm 0 . 1 8 }$ </td><td> $0 . 4 4 \pm 0 . 2 2 \ 0 . 9 1 \pm 0 . 0 2$   $0 . 3 7 \pm 0 . 1 9 0 . 8 6 \pm 0 . 0 4$   $\mathbf { 0 . 5 3 \pm 0 . 1 6 0 . 9 1 \pm 0 . 0 1 }$ </td></tr><tr><td></td><td>Every 3rd frame</td><td></td></tr><tr><td>Linear Optical Flow MedGS</td><td> $3 0 . 4 4 \pm 1 . 9 3$   $0 . 5 2 \pm 0 . 2 5$   $2 6 . 9 9 \pm 1 . 8 2$   $0 . 4 3 \pm 0 . 2 2$   ${ \bf 3 1 . 5 0 \pm 1 . 8 2 }$   ${ \bf 0 . 6 2 \pm 0 . 1 7 }$ </td><td> $0 . 3 9 \pm 0 . 1 9 0 . 8 8 \pm 0 . 0 2$   $0 . 3 0 \pm 0 . 1 6 0 . 8 2 \pm 0 . 0 3$   $\mathbf { 0 . 4 6 \pm 0 . 1 5 0 . 8 8 \pm 0 . 0 2 }$ </td></tr><tr><td></td><td>Every 5th frame</td><td></td></tr><tr><td>Linear Optical Flow</td><td> $2 8 . 1 0 \pm 2 . 1 4$   $0 . 4 4 \pm 0 . 2 2$   $2 5 . 0 8 \pm 1 . 5 6$   $0 . 3 3 \pm 0 . 1 9$ </td><td> $0 . 3 0 \pm 0 . 1 6 \ 0 . 8 4 \pm 0 . 0 4$   $0 . 2 1 \pm 0 . 1 4 0 . 7 8 \pm 0 . 0 3$ </td></tr></table>

For mesh reconstruction, we consider (i) ISO-Surface using the marching cubes algorithm [29] on voxelized masks, (ii) Poisson Surface Reconstruction, which fits smooth surfaces by solving a Poisson equation on input point clouds [20], and (iii) FUNSR, a recent implicit neural representation method reconstructing signed distance functions under geometric constraints [4].

## 5.3. Implementation Details

MedGS is implemented in PyTorch with CUDA acceleration and trained on an NVIDIA A100 GPU with 40GB of VRAM memory. Each experiment begins with 100,000 Folded-Gaussian components, adaptively densified throughout training.

For interpolation tasks, we found that a relatively high polynomial degree of the f function in Eq. (3) is beneficial, in our experiments, degree 7 is chosen to simplify hyperparameter tuning, though a grid search can optimize this if desired.

For mesh reconstruction, a polynomial function f of degree 2 suffices to model smooth anatomical variations. In this context, In-Between Frame Regularization is not applied, as segmentation masks are strictly binary and linear interpolation between binary frames would produce invalid grayscale values. Training typically takes about 20 minutes per object, whereas rendering and mesh generation require only a few seconds.

## 5.4. Evaluation Metrics

We adopt task-specific metrics to evaluate interpolation and mesh reconstruction performance.

For intensity-based volumes (e.g., Ultrasound, MRI), we use PSNR and SSIM to assess voxel intensity fidelity, and Dice Coefficient and IoU to measure overlap and structural consistency with the ground truth.

For meshes derived from segmentation masks, we report Chamfer Distance (average point-wise distance), Hausdorff Distance (maximum surface deviation), and HD95 (95thpercentile deviation), capturing both typical and worst-case geometric errors.

## 5.5. Results

As we have already mentioned, we evaluate MedGS against classical and neural baselines on both interpolation and mesh reconstruction tasks. Results are presented quantitatively in terms of accuracy, and qualitatively via visual comparisons of reconstructed slices and meshes. These experiments highlight the advantages of our Gaussian-based formulation in preserving anatomical detail.

Interpolation Interpolation performance is assessed using a leave-frame-out strategy on MRI volumes. Each dataset is sub-sampled in two ways: (i) every 2nd frame, and (ii) every 3rd frame for training, while the held-out frames serve as ground truth. During training, only selected frames are used to fit MedGS or baseline models. Subsequently, interpolated frames are generated at the temporal positions of the held-out slices and compared against the ground truth. This setup simulates realistic scenarios of sparsely sampled volumetric data and enables rigorous interpolation quality evaluation.

Table 2. Mesh reconstruction results on the Prostate Ultrasound validation dataset (Specimens 65â72). Metrics reported include Chamfer Distance (CD), Hausdorff Distance (HD), and 95th percentile Hausdorff Distance (HD95). Lower values indicate better performance. Boldface highlights the best result in each column.
<table><tr><td rowspan="2">Method</td><td colspan="6">Specimen ID  $\operatorname { A v g } .$ </td></tr><tr><td>65 66</td><td>67 68 Chamfer Distance (CD â)</td><td></td><td>3 69 70</td><td>71 72</td></tr><tr><td>FUNSR 0.239 0.216 0.227 0.217 0.209 0.249 0.230 0.238 0.228 Poisson 0.235 0.204 0.186 0.198 0.198 0.225 0.212 0.233 0.211 MedGS 0.194 0.207 0.179 0.177 0.187 0.212 0.208 0.261 0.203</td><td>Baseline 0.248 0.206 0.216 0.214 0.219 0.236 0.238 0.243 0.228</td><td colspan="4">Hausdorff Distance (HD â)</td></tr><tr><td>Baseline 0.858 0.648 0.747 0.762 0.735 0.754 0.753 0.781 0.755 FUNSR 0.745 1.089 1.059 0.694 0.731 0.826 0.766 0.771 0.835 Poisson 0.736 2.353 0.654 0.600 0.768 0.734 0.722 0.789 0.920 MedGS</td><td></td><td colspan="4">0.725 0.574 0.598 0.637 0.657 0.717 0.792 0.760 0.827</td></tr><tr><td>FUNSR Poisson</td><td>95th Percentile Hausdorff Distance (HD95 â) Baseline 0.425 0.353 0.374 0.369 0.378 0.411 0.412 0.423 0.390 0.421 0.384 0.393 0.374 0.374 0.426 0.411 0.419 0.400 0.397 0.361 0.334 0.345 0.355 0.400 0.388 0.409 0.374 MedGS</td><td colspan="4">0.365 0.350 0.325 0.332 0.350 0.386 0.384 0.427 0.365</td></tr></table>

Table 1 reports performance using PSNR, SSIM, Dice Coefficient, and IoU metrics. Baselines such as linear interpolation and optical flow struggle to preserve fine anatomical boundaries, often producing blurred or ghosted frames. In contrast, MedGS consistently achieves superior scores across all metrics and modalities (see Figure 6 and 7 for qualitative examples).

Mesh Reconstruction For mesh reconstruction, we generate 3D surfaces from segmentation masks and compare them against ground truth meshes derived from manual annotations. All methods are trained using ground truth segmentation masks to ensure fairness. For ISO-Surface extraction and MedGS, meshes are obtained directly from the masks. In contrast, Poisson reconstruction and FUNSR require point cloud inputs, which we derive from the ground truth masks prior to training.

Table 2 reports reconstruction results on the Prostate Ultrasound validation dataset. ISO-Surface extraction yields jagged, noisy meshes, while Poisson reconstruction misses fine anatomical details. FUNSR achieves competitive fidelity, but MedGS obtains the lowest Chamfer and Hausdorff distances, indicating superior geometric accuracy and anatomical preservation. However, due to the approximate nature of ground truth meshes, numerical metrics do not always correlate with visual quality, as some methods achieve favorable scores despite producing visibly inferior surfaces. Therefore, qualitative evaluation remains essential. Figure 5 demonstrates that, unlike classical methods which produce noisy or overly smoothed surfaces, MedGS preserves fine structures while maintaining mesh consistency.

For the same reason, we present only qualitative results for mesh reconstruction on the TotalSegmentator MRI Dataset. Figure 2 compares various methods, highlighting the high-quality results achieved by MedGS. For kidney and heart reconstructions, FUNSR produces comparable results; however, due to low-resolution sampling, it fails to generate smooth lung surfaces. A similar limitation is observed with Poisson reconstruction. In contrast, meshes obtained through ISO-Surface extraction exhibit a pronounced stepped (staircase-like) appearance, reflecting the discrete nature of the underlying voxel representation.

Table 3. Ablation study of MedGS on MRI brain interpolation using every 2nd frame. Reported metrics include PSNR, Dice Coefficient, and IoU, with higher values indicating better performance. Boldface highlights the best result in each column.
<table><tr><td>Variant</td><td>PSNRâ Dice Coeff. â IoU â</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { i n t e r p } } ^ { \ast }$   $3 2 . 6 8 \pm 2 . 1 7$ </td></tr><tr><td> $0 . 6 6 \pm 0 . 1 8$   $0 . 5 1 \pm 0 . 1 5$   $3 3 . 1 3 \pm 1 . 6 7$   $0 . 6 3 \pm 0 . 2 2$   $0 . 4 9 \pm 0 . 1 9$ </td></tr><tr><td>w/o  $\mathcal { L } _ { \sigma }$  w/o</td></tr><tr><td> $\mathcal { L } _ { \mathrm { i n t e r p } } ^ { \ast }$  &amp; w/o  $\scriptstyle { \mathcal { L } } _ { \sigma }$   $3 2 . 7 6 \pm 2 . 0 9$   $0 . 6 6 \pm 0 . 1 8$   $0 . 5 1 \pm 0 . 1 5$ </td></tr><tr><td>Full model (MedGS)  $3 3 . 5 2 \pm 1 . 8 7$   ${ \bf 0 . 6 7 \pm 0 . 1 8 }$   ${ \bf 0 . 5 2 \pm 0 . 1 6 }$ </td></tr></table>

Ablation Study The ablation study is conducted on the interpolation task using MRI brain volumes with a leaveframe-out strategy, where every 2nd frame is omitted during training. We assess the impact of removing key components of MedGS, specifically the interpolation regularization and the temporal spread constraint. Table 3 summarizes the results. Removing interpolation regularization leads to a modest decline in PSNR, while omitting the sigma regularization primarily reduces Dice Coefficient and IoU scores. Excluding both components further degrades overall performance compared to the full model, emphasizing that each component is essential for achieving optimal interpolation accuracy.

Summary Together, quantitative and qualitative evaluations demonstrate that MedGS outperforms both classical and neural baselines. It consistently delivers higher interpolation fidelity and more accurate mesh reconstructions, establishing itself as a robust and practical solution for multimodal 3D medical imaging.

## 6. Conclusions

In this work, we introduced MedGS, the first Gaussian Splatting-based framework specifically designed for multi-modal 3D medical imaging. By leveraging Folded-Gaussian primitives and the VeGaS formulation, our method enables robust frame interpolation and high-fidelity mesh reconstruction across ultrasound and MRI data.

The explicit and interpretable representation provided by MedGS offers a distinct advantage over implicit neural representations, allowing direct editing of anatomical structures and efficient training from sparse or noisy inputs. Extensive experiments demonstrate that MedGS consistently outperforms both classical interpolation methods and recent neural approaches, achieving superior quantitative accuracy and qualitative fidelity. Additionally, it offers practical benefits such as scalability, editability, and robustness to incomplete medical data. These attributes position MedGS as a promising tool for clinical and research applications requiring accurate reconstruction and visualization of anatomical structures.

Limitations Despite its strong performance, MedGS ultimately remains an approximation of the underlying data and may be insufficient for high-risk medical tasks where absolute accuracy is essential.

## References

[1] Tamaz Amiranashvili, David Ludke, Hongwei Bran Li, Ste- Â¨ fan Zachow, and Bjoern H Menze. Learning continuous shape priors from sparse data with neural implicit functions. Medical Image Analysis, 94:103099, 2024. 2, 3

[2] VÂ´Ä±ctor M Batlle, Jose MM Montiel, Pascal Fua, and Juan D Â´ Tardos. Lightneus: Neural surface reconstruction in en-Â´ doscopy using illumination decline. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 502â512. Springer, 2023. 3

[3] Fabian Bongratz, Anne-Marie Rickmann, and Christian Wachinger. Neural deformation fields for template-based reconstruction of cortical surfaces from mri. Medical Image Analysis, 93:103093, 2024. 3

[4] Hongbo Chen, Logiraj Kumaralingam, Shuhang Zhang, Sheng Song, Fayi Zhang, Haibin Zhang, Thanh-Tu Pham, Kumaradevan Punithakumar, Edmond HM Lou, Yuyao Zhang, et al. Neural implicit surface reconstruction of freehand 3d ultrasound volume with geometric constraints. Medical Image Analysis, 98:103305, 2024. 3, 7

[5] Matija Ciganovic, Firat Ozdemir, Fabien Pean, Philipp Fuernstahl, Christine Tanner, and Orcun Goksel. Registration of 3d freehand ultrasound to a bone model for orthopedic procedures of the forearm. International journal of computer assisted radiology and surgery, 13(6):827â836, 2018. 2

[6] Rodrigo Santa Cruz, Leo Lebrat, Pierrick Bourgeat, Clinton Fookes, Jurgen Fripp, and Olivier Salvado. Deepcsr: A 3d deep learning approach for cortical surface reconstruction. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 806â815, 2021. 2, 3

[7] Tugba Akinci DâAntonoli, Lucas K Berger, Ashraya K Indrakanti, Nathan Vishwanathan, Jakob WeiÃ, Matthias Jung, Zeynep Berkarda, Alexander Rau, Marco Reisert, Thomas Kustner, et al. Totalsegmentator mri: Robust sequence- Â¨ independent segmentation of multiple anatomic structures in mri. arXiv preprint arXiv:2405.19492, 2024. 6

[8] Mark C Eid, Pak-Hei Yeung, Madeleine K Wyburd, Joao F Ë Henriques, and Ana IL Namburete. Rapidvol: rapid reconstruction of 3d ultrasound volumes from sensorless 2d scans. In 2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI), pages 1â5. IEEE, 2025. 2, 3

[9] Anis Farshian, Markus Gotz, Gabriele Cavallaro, Charlotte Â¨ Debus, Matthias NieÃner, Jon Atli Benediktsson, and Achim Â´ Streit. Deep-learning-based 3-d surface reconstructionâa survey. Proceedings of the IEEE, 111(11):1464â1501, 2023. 2

[10] Michael Figl, Rainer Hoffmann, Marcus Kaar, and Johann Hummel. Deformable registration of 3d ultrasound volumes using automatic landmark generation. PloS one, 14 (3):e0213004, 2019. 2

[11] FrancÂ¸ois Gaits, Nicolas Mellado, and Adrian Basarab. Ultrasound volume reconstruction from 2d freehand acquisitions using neural implicit representations. In 2024 IEEE International Symposium on Biomedical Imaging (ISBI), pages 1â5. IEEE, 2024. 2, 3

[12] Xiangjun Gao, Xiaoyu Li, Yiyu Zhuang, Qi Zhang, Wenbo Hu, Chaopeng Zhang, Yao Yao, Ying Shan, and Long Quan. Mani-gs: Gaussian splatting manipulation with triangular mesh. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21392â21402, 2025. 4

[13] Karthik Gopinath, Christian Desrosiers, and Herve Lombaert. Segrecon: Learning joint brain surface reconstruction and segmentation from images. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 650â659. Springer, 2021. 3

[14] Yudong Guan, Yan Li, Ying Wang, and Hong Zhang. Study of surface reconstruction based on contours of ct image slices. In 2009 International Workshop on Intelligent Systems and Applications, pages 1â4. IEEE, 2009. 2

[15] Antoine Guedon and Vincent Lepetit. Sugar: Surface- Â´ aligned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5354â5363, 2024. 4

[16] M Hoffmann, B Billot, D Greve, J Iglesias, B Fischl, and A Dalca. Synthmorph: Learning image registration without images. IEEE Trans. Med. Imaging, 10, 2021. 2

[17] Mingjie Jiang and Bernard Chiu. A dual-stream centerlineguided network for segmentation of the common and internal carotid arteries from 3d ultrasound images. IEEE Transactions on Medical Imaging, 42(9):2690â2705, 2023. 2

[18] Zhongliang Jiang, Yue Zhou, Dongliang Cao, and Nassir Navab. Defcor-net: Physics-aware ultrasound deformation correction. Medical Image Analysis, 90:102923, 2023. 2

[19] Adam Kania, Marko Mihajlovic, Sergey Prokudin, Jacek Tabor, and PrzemysÅaw Spurek. Fresh: Frequency shifting for accelerated neural representation learning. In The Thirteenth International Conference on Learning Representations. 2

[20] Michael Kazhdan, Matthew Bolitho, and Hugues Hoppe. Poisson surface reconstruction. In Proceedings of the fourth Eurographics symposium on Geometry processing, 2006. 7

[21] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time

radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023. 2, 4, 5

[22] William Kerr, Philip Rowe, and Stephen Gareth Pierce. Accurate 3d reconstruction of bony surfaces using ultrasonic synthetic aperture techniques for robotic knee arthroplasty. Computerized Medical Imaging and Graphics, 58:23â32, 2017. 3

[23] Fabian Laumer, Mounir Amrani, Laura Manduchi, Ami Beuret, Lena Rubi, Alina Dubatovka, Christian M Matter, and Joachim M Buhmann. Weakly supervised inference of personalized heart meshes based on echocardiography videos. Medical image analysis, 83:102653, 2023. 3

[24] Baorui Ma, Zhizhong Han, Yu-Shen Liu, and Matthias Zwicker. Neural-pull: Learning signed distance functions from point clouds by learning to pull space onto surfaces. arXiv preprint arXiv:2011.13495, 2020. 3

[25] Qiang Ma, Liu Li, Emma C Robinson, Bernhard Kainz, Daniel Rueckert, and Amir Alansary. Cortexode: Learning cortical surface reconstruction by neural odes. IEEE Transactions on Medical Imaging, 42(2):430â443, 2022. 3

[26] Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. Occupancy networks: Learning 3d reconstruction in function space. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4460â4470, 2019. 3

[27] Farhan Mohamed and C Vei Siang. A survey on 3d ultrasound reconstruction techniques. Artificial IntelligenceâApplications in Medicine and Biology, pages 73â92, 2019. 2

[28] Megumi Nakao, Fei Tong, Mitsuhiro Nakamura, and Tetsuya Matsuda. Image-to-graph convolutional network for deformable shape reconstruction from a single projection image. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 259â268. Springer, 2021. 3

[29] Timothy S Newman and Hong Yi. A survey of the marching cubes algorithm. Computers & Graphics, 30(5):854â879, 2006. 2, 5, 7

[30] Duc V Nguyen, Quang N Vo, Lawrence H Le, and Edmond HM Lou. Validation of 3d surface reconstruction of vertebrae and spinal column using 3d ultrasound dataâa pilot study. Medical engineering & physics, 37(2):239â244, 2015. 2, 3

[31] Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. Deepsdf: Learning continuous signed distance functions for shape representation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 165â174, 2019. 2, 3

[32] Radu Alexandru Rosu and Sven Behnke. Permutosdf: Fast multi-view reconstruction with implicit surfaces using permutohedral lattices. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2

[33] Jorg Sander, Bob D de Vos, Steffen Bruns, Nils Planken, Â¨ Max A Viergever, Tim Leiner, and Ivana Isgum. Reconstruc- Ë tion and completion of high-resolution 3d cardiac shapes using anisotropic cmri segmentations and continuous implicit neural representations. Computers in Biology and Medicine, 164:107266, 2023. 2, 3

[34] Rodrigo Santa Cruz, Leo Lebrat, Darren Fu, Pierrick Â´ Bourgeat, Jurgen Fripp, Clinton Fookes, and Olivier Salvado. Corticalflow++: Boosting cortical surface reconstruction accuracy, regularity, and interoperability. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 496â505. Springer, 2022. 3

[35] Vincent Sitzmann, Julien Martel, Alexander Bergman, David Lindell, and Gordon Wetzstein. Implicit neural representations with periodic activation functions. Advances in neural information processing systems, 33:7462â7473, 2020. 2

[36] Weronika Smolak-Dyzewska, Dawid Malarz, Kornel Howil, Ë Jan Kaczmarczyk, Marcin Mazur, and PrzemysÅaw Spurek. Vegas: Video gaussian splatting. arXiv e-prints, pages arXivâ2411, 2024. 2, 4, 5

[37] Bowen Song, Liyue Shen, and Lei Xing. Piner: Priorinformed implicit neural representation learning for test-time adaptation in sparse-view ct reconstruction. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 1928â1938, 2023. 2, 3

[38] Yordanka Velikova, Mohammad Farid Azampour, Walter Simson, Marco Esposito, and Nassir Navab. Implicit neural representations for breathing-compensated volume reconstruction in robotic ultrasound. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 1316â1322. IEEE, 2024. 2

[39] Joanna Waczynska, Piotr Borycki, SÅawomir Tadeja, Jacek Â´ Tabor, and PrzemysÅaw Spurek. Games: Mesh-based adapting and modification of gaussian splatting. arXiv preprint arXiv:2402.01459, 2024. 4

[40] Joanna Waczynska, Tomasz Szczepanik, Piotr Borycki, Â´ SÅawomir Tadeja, Thomas Bohne, and PrzemysÅaw Spurek. Â´ Mirage: Editable 2d images using gaussian splatting. arXiv preprint arXiv:2410.01521, 2024. 2, 6

[41] Haiqiao Wang, Hong Wu, Zhuoyuan Wang, Peiyan Yue, Dong Ni, Pheng-Ann Heng, and Yi Wang. A narrative review of image processing techniques related to prostate ultrasound. Ultrasound in Medicine & Biology, 51(2):189â209, 2025. 2

[42] Zijie Wang, Megumi Nakao, Mitsuhiro Nakamura, and Tetsuya Matsuda. Shape reconstruction for abdominal organs based on a graph convolutional network. In 2021 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), pages 2960â2963. IEEE, 2021. 3

[43] David Wiesner, Julian Suk, Sven Dummer, Tereza Necasov Ë a,Â´ VladimÂ´Ä±r Ulman, David Svoboda, and Jelmer M Wolterink. Generative modeling of living cells with so (3)-equivariant implicit neural representations. Medical image analysis, 91: 102991, 2024. 2, 3

[44] Magdalena Wysocki, Mohammad Farid Azampour, Christine Eilers, Benjamin Busam, Mehrdad Salehi, and Nassir Navab. Ultra-nerf: Neural radiance fields for ultrasound imaging. In Medical Imaging with Deep Learning, pages 382â401. PMLR, 2024. 2, 3

[45] Junshen Xu, Daniel Moyer, Borjan Gagoski, Juan Eugenio Iglesias, P Ellen Grant, Polina Golland, and Elfar Adal-

steinsson. Nesvor: implicit neural representation for slice-tovolume reconstruction in mri. IEEE transactions on medical imaging, 42(6):1707â1719, 2023. 2, 3

[46] Pak-Hei Yeung, Linde S Hesse, Moska Aliasi, Monique C Haak, Weidi Xie, Ana IL Namburete, INTERGROWTH 21st Consortium, et al. Sensorless volumetric reconstruction of fetal brain freehand ultrasound scans with deep implicit representation. Medical Image Analysis, 94:103147, 2024. 2, 3

[47] Wayne Y Zhang, Robert N Rohling, and Dinesh K Pai. Surface extraction with a three-dimensional freehand ultrasound system. Ultrasound in medicine & biology, 30(11):1461â 1473, 2004. 2, 3

[48] Youwei Zhang, Robert Rohling, and Dinesh K Pai. Direct surface extraction from 3D freehand ultrasound images. IEEE, 2002. 3

[49] Haoyin Zhou and Jayender Jagadeesan. Real-time dense reconstruction of tissue surface from stereo optical video. IEEE transactions on medical imaging, 39(2):400â412, 2019. 3

[50] S Kevin Zhou, Daniel Rueckert, and Gabor Fichtinger. Handbook of medical image computing and computer assisted intervention. Academic Press, 2019. 3