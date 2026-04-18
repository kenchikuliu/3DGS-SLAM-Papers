<!-- page 1 -->
LOD-GS: Level-of-Detail-Sensitive 3D Gaussian Splatting for
Detail Conserved Anti-Aliasing
Zhenya Yang1 Bingchen Gong2 Kai Chen1
1 The Chinese University of Hong Kong
2 Ecole Polytechnique
Abstract
Despite the advancements in quality and efficiency
achieved by 3D Gaussian Splatting (3DGS) in 3D scene
rendering, aliasing artifacts remain a persistent challenge.
Existing approaches primarily rely on low-pass filtering to
mitigate aliasing. However, these methods are not sensi-
tive to the sampling rate, often resulting in under-filtering
and over-smoothing renderings. To address this limitation,
we propose LOD-GS, a Level-of-Detail-sensitive filtering
framework for Gaussian Splatting, which dynamically pre-
dicts the optimal filtering strength for each 3D Gaussian
primitive. Specifically, we introduce a set of basis func-
tions to each Gaussian, which take the sampling rate as
input to model appearance variations, enabling sampling-
rate-sensitive filtering.
These basis function parameters
are jointly optimized with the 3D Gaussian in an end-to-
end manner. The sampling rate is influenced by both fo-
cal length and camera distance. However, existing meth-
ods and datasets rely solely on down-sampling to simu-
late focal length changes for anti-aliasing evaluation, over-
looking the impact of camera distance. To enable a more
comprehensive assessment, we introduce a new synthetic
dataset featuring objects rendered at varying camera dis-
tances. Extensive experiments on both public datasets and
our newly collected dataset demonstrate that our method
achieves SOTA rendering quality while effectively elimi-
nating aliasing.
The code and dataset are available at
https://github.com/Huster-YZY/LOD-GS.
1. Introduction
Novel view synthesis (NVS) plays a crucial role in the fields
of computer vision and computer graphics. Advanced NVS
techniques significantly enhance applications in virtual re-
ality, digital modeling, and embodied AI. A notable mile-
stone in NVS is the Neural Radiance Field (NeRF) [20],
which represents a 3D scene using a multi-layer perceptron
(MLP) and optimizes this MLP through volume ray march-
ing [5, 14, 19] and gradient descent based on multi-view
inputs. In comparison to NeRF [20], recent advancements
a) 3DGS
b) Mip-Splatting
c) LOD-GS
Aliasing
Free
Detail
Conserved
Under
Filtering
Over
Smoothing
Dilation
Erosion
Figure 1. (a) 3DGS treats a 3D Gaussian primitive (in blue) uni-
formly across different views and applies a dilation operator (in
red) before rendering.
(b) Mip-Splatting applies the fixed 3D
smoothing filter (in yellow) to the primitive across all views. (c)
LOD-GS applies different filters (in yellow and orange) based on
the sampling rates of the views. Mip-Splatting utilizes 2D Mip fil-
ter (in red) and LOD-GS uses EWA filter (in red) before rendering.
The lack of filtering and the use of a dilation operator in 3DGS re-
sult in erosion and dilation effects. The fixed 3D smoothing filter
in Mip-Splatting leads to over-smoothing and under-filtering.
in 3D Gaussian Splatting (3DGS) [11] offer an alternative
approach by representing a 3D scene with a collection of
3D Gaussians. This method benefits from an efficiently im-
plemented CUDA rasterizer, which facilitates real-time ren-
dering and efficient training processes. These advantages
position 3DGS as a competitive alternative to NeRF.
However, 3DGS also faces aliasing problems [7] like
NeRF. 3DGS is able to achieve good novel view synthe-
sis performance when the training and testing views share
roughly the same sampling rate to the observed scene, de-
fined as the ratio of focal length to camera distance. So
the aliasing problem of 3DGS is unobvious because exist-
ing datasets usually capture a scene from nearly the same
distance with the same focal length.
For 3DGS models
trained on these single-scale datasets, they tend to generate
degraded results or aliased renderings when test views zoom
in or out significantly. The main reason for this problem
is that the 3DGS model is fixed after training. It does not
arXiv:2507.00554v3  [cs.CV]  30 Aug 2025

<!-- page 2 -->
adjust its appearance according to the sampling rate. This
limitation leads to the aliasing and degraded results.
In this paper, we are inspired by the Mipmap tech-
nique [32], which is used in computer graphics rendering
pipelines to tackle the aliasing. Mipmap involves a col-
lection of progressively downsampled textures, enabling
the program to choose the appropriate resolution based on
the camera’s sampling rate——a technique known as pre-
filtering. This method is grounded in the Nyquist-Shannon
Sampling Theorem [22, 25] and its concept can be gener-
alized as Level of Detail (LOD) [8, 12, 18, 31], which in-
dicates that the appearance of an object changes with the
sampling rate for the scene.
We adopt the Mipmap technique into the 3D Gaussian
Splatting framework [11] to represent the pre-filtered radi-
ance field with different levels of detail using multiscale im-
ages as input. Because the vanilla 3DGS is not sensitive to
the sampling rate, training it with multi-scale images can
lead to ambiguity in optimization, making rendering results
blur and lack of detail. In this paper, we propose a level-of-
detail-sensitive 3DGS framework, named LOD-GS, to ef-
fectively sense the change in sampling rate and train the pre-
filtered radiance field from pre-filtered images. We propose
to add a set of basis functions on each Gaussian primitive to
learn the appearance change across different sampling rates.
These basis functions take the sampling rate as input and
predict the filter size and appearance change for each prim-
itive as illustrated in Figure 1 and Figure 3. The filtered
primitive will be splatted into 2D screen space and perform
the Elliptical Weighted Average (EWA) filtering [41] to fur-
ther improve anti-aliasing ability of our method. Besides
the framework, we re-render the synthetic dataset used in
NeRF [20] from three different camera distances to simu-
late the changes in sampling rate caused by varying cam-
era distances. The evaluation is carried out on both public
datasets [2, 3, 20] and our newly collected dataset. Exper-
iment results demonstrate that our method can disentangle
training views with different sampling rates and learn a pre-
filtered radiance field from these views, achieving detail-
conserved and aliasing-free rendering at the same time.
Our contributions are as follows:
• We
introduce
LOD-GS,
a
Level-of-Detail-sensitive
Gaussian Splatting framework that enables effective anti-
aliasing while preserving fine details in rendering results.
• Our proposed LOD-GS eliminates the ambiguity in train-
ing caused by inputs with different sampling rates, allow-
ing it to effectively learn a pre-filtered radiance field from
pre-filtered images.
• We extend the original NeRF Synthetic Dataset [20] by
incorporating rendering views from different camera dis-
tances, providing a more comprehensive evaluation of
zoom-in and zoom-out effects in neural radiance fields.
3DGS
Mip-Splatting
Ours
Analytic-Splatting
Ground-Truth
Figure 2. Comparison of Detail Reconstruction. All methods
are trained on inputs with varying sampling rates. In comparing
the reconstruction results of different methods, only our LOD-GS
successfully reconstructs the intricate texture of the silk ribbon.
Other methods share the smoothing problem to different extents.
2. Related Work
Novel View Synthesis
Novel View Synthesis (NVS) is a
fundamental vision task aimed at generating plausible ren-
derings from new camera positions based on a set of input
images and their corresponding camera positions. The in-
troduction of Neural Radiance Fields (NeRF) [20] has sig-
nificantly changed the approach to solving the NVS task.
NeRF formulates Novel View Synthesis as a volume ren-
dering and optimization problem.
Nearly all subsequent
works adhere to this framework. However, due to the fre-
quent querying of the MLP during training and inference,
vanilla NeRF [20] is quite slow, requiring over ten hours for
training per scene and dozens of seconds to render a new
view. Following works have replaced the full implicit rep-
resentation of NeRF with feature grid-based representations
to accelerate training and inference speeds [6, 9, 17, 21, 27].
Some approaches also explore the point-based neural repre-
sentations [13, 23, 28, 33, 35, 39]. The emergence of 3D
Gaussian Splatting [11], which represents the scene as a
collection of 3D Gaussians, has removed the MLP from the
radiance field representation. This specially designed MLP-
free framework, combined with efficient implementation,
allows 3DGS to achieve real-time rendering speeds and per-
form efficient training. Due to these advantages, 3DGS is
beginning to replace NeRF in many applications.
How-
ever, 3DGS is struggling to tackle the aliasing, this paper
focuses on enhancing the anti-aliasing capability of 3DGS
to achieve aliasing-free and detail conserved rendering.

<!-- page 3 -->
Projection
Differential
Rasterization
Image
EWA filter
3D Gaussians
Filtered 3D Gaussians
Operation Flow
Gradient Flow
Learnable 3D Filters
…
…
f/d
(Δs, Δα)
f/d
(Δs, Δα)
f/d
(Δs, Δα)
Gaussian Function
Camera Sampling Rates
Camera View
+
Figure 3. Overview of our pipeline. During rendering, the sampling rate for each Gaussian primitive is sent to a learnable module that
predicts the appropriate 3D filters for them. The filtered 3D Gaussians are then projected into 2D space and further processed using EWA
filter before rasterization. The learnable 3D filters and the 3D Gaussians are jointly optimized end-to-end with image supervision.
Anti-Aliasing of Neural Radiance Field
Neural render-
ing methods integrate pre-filtering techniques to mitigate
aliasing [2–4, 10, 36, 40]. Mip-NeRF [2] is the first work
to address the aliasing problem in NeRF by proposing In-
tegrated Positional Encoding (IPE), which filters out high-
frequency components when the camera’s sampling rate is
low. Following works [3, 10] speed up the training of Mip-
NeRF and extend it to handle unbounded scenes. Due to the
differences in rendering methods between NeRF and 3DGS,
the anti-aliasing strategy used in NeRF cannot be directly
applied to 3DGS. Several methods have recently been pro-
posed to address the aliasing problem in Gaussian Splat-
ting [15, 16, 24, 26, 34, 36]. Mip-Splatting [36] introduces
the 3D smoothing filter and the 2D mip-filter to remove
high-frequency components from the Gaussians. Mipmap-
GS [15] utilizes a two stage training and generated pseudo
ground truth to adapt to a specific resolution.
Analytic-
Splatting [16] proposes an approximate method to compute
the integration of Gaussian, effectively tackling the alias-
ing problem. However, all these methods have their lim-
itations. Mipmap-GS needs re-training once the sampling
rate changes. The introduction of integration in Analytic-
Splatting slows down its training and rendering. The 3D
filter in Mip-Splatting remains fixed after training, regard-
less of the camera’s sampling rate. Additionally, the 2D
Mip filter cannot sense the existence of the 3D filter, po-
tentially resulting in inappropriate 2D filtering degree. All
these factors contribute to the problem of under-filtering and
over-smoothing, as illustrated in Figure 2. To solve these
problems, we propose a filtering strategy which dynami-
cally adjusts the filtering degree and learns this adjustment
from data in an end-to-end manner.
3. Preliminaries
3.1. 3D Gaussian Splatting
Kerbl et al. [11] utilize learnable 3D Gaussian primitives to
represent 3D scenes and render different views using a dif-
ferentiable volume splatting rasterizer. In 3DGS, each 3D
Gaussian primitive is parameterized using a 3D covariance
matrix Σ and a distribution center pk:
G(p) = exp

−1
2(p −pk)⊤Σ−1(p −pk)

(1)
During optimization, the covariance matrix Σ is factorized
into a scaling matrix S and a rotation matrix R as Σ =
RSS⊤R⊤to ensure its positive semidefiniteness. To obtain
the rendering results of 3D Gaussians from a specific view,
the 3D Gaussian is first projected to a 2D splat in screen
space using the view matrix W and an affine approximated
projection matrix J as illustrated in [41]:
Σ′ = JWΣW⊤J⊤
(2)
By removing the third row and column of Σ′, we obtain a
2 × 2 matrix, which represents the covariance matrix Σ2D
of the 2D splat G2D. Finally, the color of each pixel x can
be computed using volumetric alpha blending as follows:
c(x) =
K
X
k=1
ckαkG2D
k (x)
k−1
Y
j=1
(1 −αjG2D
j
(x))
(3)
where k is the index of the Gaussian primitives covering
the current pixel, αk denotes the alpha values, and ck repre-
sents the view-dependent appearance modeled using Spher-
ical Harmonics. All attributes of the 3D Gaussian primitives
(p, S, R, α, c) are optimized using the photometric loss be-
tween the rendered images and the ground-truth images.
3.2. Nyquist-Shannon Sampling Theorem
Many signals initially exist in analog form. Sampling con-
verts these continuous signals into a discrete format suitable
for processing, storage, and transmission by digital devices.
However, improper sampling can cause distortion known as
aliasing. The Nyquist-Shannon sampling theorem states
that the sampling rate ν must be at least twice the bandwidth
of the signal to avoid aliasing. Based on this theorem, there
are two common approaches to address aliasing: one is to
increase the sampling rate, which inevitably raises the com-
putational workload; the other is to apply bandpass filters

<!-- page 4 -->
to the signals to eliminate frequencies higher than ν/2. The
frequency ν/2 is known as the Nyquist frequency.
The rendering process of 3D Gaussian Splatting in-
volves sampling.
It uses discrete pixels to sample 3D
Gaussians, ultimately producing a 2D rendering result from
this 3D representation.
Consequently, the rendering of
3D Gaussian Splatting also faces the challenge of alias-
ing. Corresponding to the two approaches mentioned ear-
lier, the first method to address aliasing in 3DGS render-
ing is Super Sampling [29]. This technique renders im-
ages at a higher resolution—effectively increasing the sam-
pling rate—and then pools the high-resolution results to
generate anti-aliased low-resolution images. However, the
high resolution used in Super Sampling leads to additional
computational burden, resulting in slower rendering speeds.
The second approach applies filtering to 3D Gaussians be-
fore rendering, ensuring their frequency remains below the
Nyquist limit. We adopt this method for efficient rendering.
However, excessive filtering can cause signal loss, leading
to overly smooth results that lack fine details. In Section 4,
we will explain in detail how our method selects the appro-
priate filter for each Gaussian primitive.
4. Method
In this part, we first introduce our Level-of-Detail-Sensitive
3D Filter in Section 4.1, which takes the sampling rate as
input and predicts the filters for 3D Gaussians. This de-
sign helps 3DGS disentangle the inputs of different sam-
pling rates to better learn a pre-filtered radiance field from
per-filtered images without ambiguity. Then we illustrate
the EWA filter technique in Section 4.2, which is employed
to enhance the anti-aliasing capabilities of our method. The
overview of our method is illustrated in Figure 3.
4.1. Level-of-Detail-Sensitive 3D Filter
Applying zoom-in and zoom-out operations on scenes re-
constructed by 3DGS can lead to noticeable visual artifacts,
including erosion and dilation [36]. To solve these artifacts,
Mip-Splatting proposes 3D Smoothing Filter as follows :
Gk(x) =
s
|Σk|
|Σk +
s
ˆ
νk |e−1
2 (x−pk)T (Σk+ s
ˆ
νk )(x−pk)
(4)
where the hyper parameter s is used to control the filter size
and ˆνk is the maximal sampling rate for the k-th primitive
which could be computed as follows:
ˆνk =
max
n∈{1,...,N}
fn
dn
(5)
where N is the number of training views, fn is the focal
length of the n-th view, and dn is the distance from the n-
th view to the current primitive. The time complexity of
Equation 5 is O(KN), where K is the number of Gaus-
sian primitives and N is the number of training views. To
reduce computational load, Mip-Splatting recomputes ˆνk
every 100 iterations. After training, the ˆν for each primi-
tive is fixed, meaning that during testing, the filter size re-
mains unchanged regardless of variations in the sampling
rate. As a result, the training process becomes dependent
on the choice of the hyperparameter s. As shown in Fig-
ure 1, Figure 2 and Figure 5, a fixed 3D smoothing filter can
lead to some textures being over-filtered while others are
not sufficiently filtered when the test camera moves closer
or farther.
In the Level of Detail (LOD) concept [8, 12, 18, 31],
an object’s texture resolution adapts based on the sampling
rate. Inspired by this idea, we design a learnable framework
that takes the sampling rates of Gaussian primitives as input
and outputs appropriate 3D filters. In other words, the filter
size of each primitive dynamically adjusts according to the
sampling rate. To minimize the increase in computational
workload of training and inference, we design a learnable
Gaussian Mixture Model (GMM) module instead of using
a MLP for each primitive to predict the suitable filter size.
The added GMM module can be expressed using the fol-
lowing equation:
F(x) =
l
X
i=1
wi exp(−(x −µi)2
2σ2
i
)
(6)
where l is the number of basis functions, ui and σi are
the distribution center and standard deviation which will be
optimized using gradient descent. The input of the GMM
module is the sampling rate, defined as: ν = 1/T = f/d,
where f is the focal length and d is the distance from camera
to the primitive. T is defined as the sampling interval. We
also add a learnable residual in opacity to model the opacity
change during the filtering, this design helps us avoid the
use of hyperparameter in Equation 4. Above all, our pro-
posed LOD sensitive filter could be represented as:
Gk(x) = e−1
2 (x−pk)T (Σk+Fs(ν))(x−pk)
(7)
where ν is the sampling rate of the Gaussian primitive.
The sampling rate from one camera to all the primitives can
be computed efficiently in O(K). This allows us to per-
form this computation in each iteration, rather than updat-
ing the sampling rate every 100 iterations, as done in Mip-
Splatting [36]. This design makes each Gaussian primitive
sensitive to changes in the sampling rate. As a result, the
scale and opacity of the primitive can be adjusted accord-
ingly. The following experimental results demonstrate that
our proposed method can better bake images from differ-
ent sampling rates into a single radiance field, generating
aliasing-free rendering results while preserving details.

<!-- page 5 -->
PSNR ↑
SSIM ↑
LPIPS ↓
Full Res. 1/2 Res. 1/4 Res. 1/8 Res. Avg. Full Res. 1/2 Res. 1/4 Res. 1/8 Res. Avg. Full Res. 1/2 Res. 1/4 Res. 1/8 Res Avg.
NeRF [20]
29.90
32.13
33.40
29.47 31.23
0.938
0.959
0.973
0.962 0.958
0.074
0.040
0.024
0.039 0.044
MipNeRF [2]
32.63
34.34
35.47
35.60 34.51
0.958
0.970
0.979
0.983 0.973
0.047
0.026
0.017
0.012 0.026
Tri-MipRF [10]
32.65
34.24
35.02
35.53 34.36
0.958
0.971
0.980
0.987 0.974
0.047
0.027
0.018
0.012 0.026
3DGS [11]
28.79
30.66
31.64
27.98 29.77
0.943
0.962
0.972
0.960 0.960
0.065
0.038
0.025
0.031 0.040
Mipmap-GS [15]
28.79
30.67
31.66
28.00 29.78
0.943
0.962
0.973
0.961 0.960
0.065
0.038
0.025
0.031 0.040
SA-GS [26]
30.80
32.67
35.06
35.77 33.58
0.956
0.969
0.980
0.985 0.973
0.056
0.032
0.020
0.014 0.031
Multiscale-3DGS[34]
33.36
27.15
21.41
17.61 24.88
0.969
0.951
0.875
0.764 0.890
0.031
0.032
0.067
0.126 0.064
Mip-Splatting [36]
32.81
34.49
35.45
35.50 34.56
0.967
0.977
0.983
0.988 0.979
0.035
0.019
0.013
0.010 0.019
Analytic-Splatting [16]
33.22
34.92
35.98
36.00 35.03
0.967
0.977
0.984
0.989 0.979
0.033
0.019
0.012
0.010 0.018
LOD-GS (ours)
32.90
34.88
36.43
37.27 35.37
0.966
0.977
0.985
0.990 0.980
0.035
0.019
0.012
0.008 0.018
Table 1. Multi-scale Training and Multi-scale Testing on the Blender dataset [20]. Our approach demonstrates state-of-the-art per-
formance across most metrics and is highly competitive with existing GS-based methods specifically designed for anti-aliasing, such as
Mipmap-GS [15], SA-GS [26], Multiscale-3DGS [34], Mip-Splatting [36], and Analytic-Splatting [16].
4.2. EWA Filtering
In the implementation of 3DGS, a dilation operator [36] is
applied to each projected Gaussian. This helps avoid small
primitives that are hard to optimize. The dilation operator
could be written as: G2D
k (x) = e−1
2 (x−pk)T (Σ2D
k
+sI)(x−pk).
This operator leads to obvious dilation artifacts when the
image resolution decreases. These artifacts are primarily
caused by the expansion of the primitive’s scale without ad-
justing its opacity. As a result, the overall energy increases,
leading to the dilation effect. The EWA Filtering [41] could
be used to alleviate this artifact:
G2D
k (x) =
s
|Σ2D
k |
|Σ2D
k
+ sI|e−1
2 (x−pk)T (Σ2D
k
+sI)(x−pk) (8)
The main difference between dilation operator and EWA
filter is the normalization term preceding the exponential
term. In LOD-GS, we use the EWA filter before the alpha
blending computation to reduce the dilation and aliasing ar-
tifact. After the EWA filtering, we perform alpha blending
to obtain the rendering result as follows:
c(x) =
K
X
k=1
ck ˆ
αkG2D
k (x)
k−1
Y
j=1
(1 −ˆαjG2D
j
(x))
(9)
ˆαk = αk + Fα(ν)
(10)
where ˆα is the opacity of the LOD-filtered Gaussian prim-
itive. After obtaining the rendering results, we compute the
image loss between these results and the ground truth. Then
we perform gradient descent to optimize all Gaussian pa-
rameters and the filter prediction module in an end-to-end
manner. This design enables the LOD filter to adjust its
filtering degree by taking the EWA filter into account, al-
lowing it to predict the most appropriate filter for achiev-
ing both aliasing-free and detail-conserved rendering. Con-
versely, the 3D filter and 2D Mip filter of Mip-Splatting op-
erate sequentially but independently, which can easily result
in over-filtering and lead to blurry rendering results.
5. Experimental Evaluation
5.1. Implementation
The LOD filtering is implemented using PyTorch and is de-
signed to be plug-and-play, allowing for easy integration
into existing 3DGS-based methods. It is implemented us-
ing GMM and we set the number of basis functions to 20
in the following experiment. We evaluate our method on
both synthetic and real-world datasets, including the Multi-
scale Synthetic Dataset from Mip-NeRF [2] and the Mip-
NeRF 360 [3] dataset. We render a multi-level synthetic
dataset for more comprehensive evaluation as illustrated in
Section 5.3. We also evaluate the generalization ability of
our method in Section 5.6 and examine the impact of the
number of basis functions used in the LOD filter in Sec-
tion 5.7. In our experiment, Mip-Splatting uses the den-
sification scheme from GOF [37], while all other meth-
ods, including LOD-GS, utilize the original densification
scheme of 3DGS. All experiments are conducted on a sin-
gle RTX3090 GPU.
5.2. Evaluation on the Multi-scale Blender Dataset
The Blender dataset introduced in the original NeRF [20]
is a synthetic dataset where all training and testing im-
ages observe the scene content from a roughly constant
distance with the same focal length, which differs sig-
nificantly from real-world captures. MipNeRF [1] intro-
duces a multi-scale Blender dataset designed to enhance
the evaluation of reconstruction accuracy and anti-aliasing
in multi-resolution scenes.
This dataset is generated by
downscaling the original dataset by factors of 2, 4, and
8, and then combining these variations. We evaluate our
LOD-GS on this dataset with several competitive meth-
ods, including NeRF-based methods (NeRF [20], MipN-
eRF [1], and Tri-MipRF [10]) and 3DGS-based methods
(Mipmap-GS [15], SA-GS [26], Multiscale-3DGS [34],
Mip-Splatting [36], and Analytic-Splatting [16]).
Fol-

<!-- page 6 -->
3DGS
Multiscale-3DGS
Mip-Splatting
Analytic-Splatting
Ours
Ground Truth
Figure 4. Qualitative comparison on our extended Blender dataset. All methods are trained across three levels: L1 (near), L2 (middle),
and L3 (far). Our method more effectively captures the details of microstructures and textures during the multi-level training.
PSNR ↑
SSIM ↑
LPIPS ↓
Near Medium
Far
Avg.
Near Medium
Far
Avg.
Near Medium
Far
Avg.
3DGS [11]
27.20
32.57
37.12 32.30 0.906
0.970
0.995 0.957 0.150
0.031
0.005 0.062
Mipmap-GS [15]
26.59
31.61
34.66 30.95 0.898
0.967
0.992 0.952 0.157
0.034
0.005 0.066
Multiscale-3DGS[34]
26.31
31.54
34.66 30.84 0.898
0.967
0.992 0.952 0.158
0.034
0.005 0.066
Mip-Splatting [36]
26.87
33.08
40.76 33.57 0.902
0.973
0.997 0.957 0.154
0.029
0.003 0.062
Analytic-Splatting [16] 27.06
33.27
40.41 33.58 0.901
0.972
0.997 0.957 0.154
0.030
0.003 0.062
LOD-GS (ours)
27.30
33.40
41.27 33.99 0.905
0.973
0.997 0.958 0.148
0.030
0.003 0.060
Table 2. Multi-level Training and Multi-level Testing on our extended Blender dataset. Our LOD-GS achieves state-of-the-art perfor-
mance in multi-level scene representation compared to vanilla 3DGS and existing anti-aliasing Gaussian splatting methods.
lowing previous works, we report three metrics: PSNR,
SSIM [30], and VGG LPIPS [38] across four resolutions
and average results, as shown in Table 1. Mip-NeRF and
Tri-MipRF exhibit stronger anti-aliasing capabilities com-
pared to vanilla NeRF. For 3DGS-based methods, Mip-
Splatting and Analytic-Splatting significantly enhance the
anti-aliasing ability of 3DGS. While Multiscale-3DGS can
achieve better performance at the original resolution, it de-
generates at lower resolutions. It is important to empha-
size that both Mip-Splatting and Analytic-Splatting require
sampling more cases from high resolutions to maintain their
performance at the original resolution. In contrast, by utiliz-
ing LOD-sensitive filter control, our method achieves rather
good performance without relying on any dataset resam-
pling trick. As shown in Table 1, our LOD-GS achieves
state-of-the-art anti-aliasing performance while maintaining
competitive rendering quality at the original resolution.
5.3. Evaluation on the Multi-level Blender Dataset
Both decreasing the focal length and increasing the cam-
era distance can change the sampling rate, thereby reducing
the resolution of the photographed object. Mip-NeRF re-
duces image resolution using image processing techniques,
attributing this change to alterations in focal length. How-
ever, in 3D applications, the most common approach to
change the sampling rate is by adjusting the camera dis-
tance, rather than the focal length. There is a difference
between adjusting the camera distance and the focal length.
To make a more comprehensive evaluation, we re-render
the synthetic dataset used in NeRF [20] from three different
camera distances. We call this extended dataset the “Multi-
level Blender Dataset” and assign Level 1 to the camera

<!-- page 7 -->
3DGS
Mip-Splatting
Ours
Ground Truth
Figure 5. Qualitative comparison on Anti-aliasing. We com-
pare the results of our method with the aliased rendering results of
3DGS and the anti-aliased results of Mip-Splatting to illustrate the
anti-aliasing capabilities of our proposed method. The positions
of the details on the original objects are marked with red boxes.
with the nearest camera distance to the object, Level 2 to
the middle camera distance, and Level 3 to the farthest dis-
tance. The results of the comparative experiments on this
dataset are presented in Table 2. As shown, 3DGS performs
well when the object is close to the camera but degrades in
performance when the camera is far from the object. Con-
versely, Mip-Splatting and Analytic-Splatting exhibit better
performance at greater distances but struggle at near dis-
tances. Only LOD-GS achieves good performance at both
near and far distances. It is crucial to note that when the
camera is positioned at a far distance, a significant portion
of the image appears white. This can lead to smaller differ-
ences in SSIM and LPIPS between various methods. In this
context, PSNR is a more appropriate metric, as it is calcu-
lated based on pixel differences.
Qualitative results.
We further conducted a qualitative
comparison of our LOD-GS with vanilla 3DGS and exist-
ing GS-based methods for the reconstruction of details. As
shown in Figure 4, both 3DGS and Multiscale-3DGS fail
to capture very fine details, resulting in noticeable discrep-
ancies in appearance compared to the ground-truth images.
Although Mip-Splatting and Analytic-Splatting retain some
texture detail, they still lose additional intricacies and pro-
duce overly blurred rendering results in certain areas. In
contrast, our method captures more details and achieves
higher rendering quality. To be more precise, as illustrated
in Figure 4, our method excels at reconstructing the small
bump on the back of the chair, the intricate texture on the
surface of the microphone, and the rope net of the ship. In
3DGS
Mip-Splatting
Ours
Ground Truth
Figure 6.
Qualitative comparison on the Mip-NeRF 360
dataset [3]. All methods are trained and tested at three downsam-
pled resolutions (1/8, 1/16, and 1/32). We recommend that readers
scale up this image and compare the region marked with red boxes
across different methods.
contrast, other methods either fail to capture these fine de-
tails or produce overly blurred results.
In addition to capturing fine details, we also evaluate the
anti-aliasing capabilities of our method. We render the ob-
ject from a considerable distance to assess this ability. The
qualitative results are illustrated in Figure 5. As shown,
3DGS tends to generate dilated boundaries and aliased re-
sults in areas with high-frequency texture. In contrast, both
Mip-Splatting and our LOD-GS effectively handle aliasing
when the camera is far from the scene, i.e., at a low sam-
pling rate. Compared to Mip-Splatting, our method can re-
construct more details, such as the inner structure of the mi-
crophone and the rope net of the ship, as shown in the third
row and the fourth row of the Figure 5.
5.4. Evaluation on the Mip-NeRF 360 Dataset
To test the performance of our method on real-world data,
we evaluate our method and perform comparisons on the
Mip-NeRF 360 Dataset [3]. Due to the high resolution of
images in the 360 Dataset and the need to perform multi-
scale training, we conduct multiscale training and testing
on three downsampled scales (1/8, 1/16, and 1/32) to reduce
the VRAM consumption. The qualitative and quantitative
experimental results are shown in Figure 6 and Table 3. As
shown in Table 3, our method achieves SOTA performance
across nearly all metrics at different resolutions.
Qualitative Results
To provide a more intuitive compar-
ison of real-world scenes, we analyze the rendering results
of different methods across various scenes, as shown in Fig-
ure 6. Compared to 3DGS and Mip-Splatting, the most no-
table distinction of our method is its ability to reconstruct
highly detailed geometry and texture following multiscale
training. For instance, in Figure 6, our method effectively

<!-- page 8 -->
PSNR ↑
SSIM ↑
LPIPS ↓
1/8 Res. 1/16 Res. 1/32 Res. Avg. 1/8 Res. 1/16 Res. 1/32 Res. Avg. 1/8 Res. 1/16 Res. 1/32 Res. Avg.
3DGS [11]
27.99
29.96
29.62
29.19 0.850
0.919
0.934
0.901 0.158
0.074
0.057
0.096
Mipmap-GS [15]
27.28
29.23
28.50
28.33 0.836
0.909
0.919
0.888 0.173
0.084
0.066
0.108
Multiscale-3DGS[34]
27.30
29.25
28.50
28.35 0.836
0.910
0.920
0.888 0.173
0.084
0.066
0.108
Mip-Splatting [36]
28.35
30.16
31.06
29.86 0.865
0.924
0.949
0.913 0.132
0.064
0.043
0.080
Analytic-Splatting [16]
28.84
30.63
31.63
30.37 0.868
0.926
0.953
0.915 0.128
0.066
0.043
0.079
LOD-GS (ours)
28.99
30.93
32.48
30.80 0.870
0.929
0.958
0.919 0.133
0.065
0.038
0.078
Table 3. Multi-scale training and Multi-scale testing on the Mip-NeRF 360 dataset [3]. All methods are trained and tested at three
downsampled resolutions (1/8, 1/16, and 1/32). Our model achieves state-of-the-art performance across most metrics in comparison.
5.0
7.5
10.0
12.5
15.0
17.5
20.0
34.6
34.8
35.0
35.2
35.4
PSNR
5.0
7.5
10.0
12.5
15.0
17.5
20.0
0.9788
0.9790
0.9792
0.9794
0.9796
SSIM
5.0
7.5
10.0
12.5
15.0
17.5
20.0
0.0184
0.0186
0.0188
0.0190
0.0192
LPIPS
Figure 7. Performance of Models Using Different Numbers of Basis Functions: We report our model’s changes in PSNR, SSIM, and
LPIPS by using 5, 10, 15, and 20 basis functions. Metrics for Mip-Splatting and Analytic-Splatting are visualized with dashed lines.
Table 4. Ablation on Multiscale
Blender dataset.
PSNR↑SSIM↑LPIPS↓
w/o LOD
34.39
0.977
0.022
w/o EWA
34.61
0.978
0.020
full model
35.37
0.980
0.018
Table 5. STMT on Multi-
level Blender dataset.
PSNR ↑Zoom
In
Trained
Scale
Zoom
Out
3DGS
23.38
35.73
37.85
Mip
23.50
35.30
39.54
Analytic 23.40
34.66
40.09
LOD-GS 24.27
35.95
41.42
reconstructs the microstructure of the plant in the garden
scene, the texture of the onion in the counter scene, and
the complex distribution of leaves in the treehill scene. All
these results demonstrate that our method achieves high-
quality rendering after multiscale training.
5.5. Ablation Study
We remove the LOD filtering and EWA filter, respec-
tively, to validate the effectiveness of these two components.
The ablation study is conducted on the Multiscale Blender
Dataset and the experiment result is presented in Table 4.
Only using the LOD filtering can achieve rather good per-
formance while utilizing EWA filter can further improve it.
Because the EWA filter can sense the change in sampling
rate to some degree, utilizing it can decrease the learning
burden of LOD filtering, thus achieving better performance
using the same number of training iterations.
5.6. Generalization on STMT
Our method is primarily designed for Multiscale Training
and Multiscale Testing (MTMT). To test the generalization
of our method, we also perform the Single-scale Training
and Multiscale Testing (STMT) on the Multi-level Blender
dataset. As shown in Table 5, our method achieves the best
performance across all scales, even if we only train it using
one scale. The experimental result demonstrates that our
method can generalize to STMT well.
5.7. Analysis of the Number of Basis Functions
We investigate the influence of the number of basis func-
tions on the model’s performance and present the experi-
mental results in Figure 7. As shown, using only 5 basis
functions allows our method to surpass both Mip-Splatting
and Analytic-Splatting in terms of PSNR and SSIM. With
an increase in the number of basis functions, our model
exhibits improved performance across PSNR, SSIM, and
LPIPS; however, this also results in higher computational
costs. This observation provides a guideline for selecting
the number of basis functions: opting for fewer functions
for efficiency or more functions for enhanced performance.
6. Conclusion
In this paper, we propose a Level-of-Detail-Sensitive Gaus-
sian Splatting method, LOD-GS, to sense and learn the fil-
tering degree changes caused by variations in the camera
sampling rate. Compared to existing methods, our approach
achieves state-of-the-art performance on anti-aliasing tasks
while preserving very fine details.
This work also re-
renders the Synthetic NeRF dataset from different camera
distances to facilitate the comprehension evaluation of the
anti-aliasing task. Experimental results also demonstrate
that LOD-GS generalize well to STMT and can achieve
rather good performance with minor additional parameters.
LOD-GS is designed for easy reproduction and integration.
It is expected to further improve the reconstruction quality
of neural radiance fields, especially when the collected im-
ages are captured at varying sampling rates.

<!-- page 9 -->
References
[1] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter
Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan.
Mip-nerf: A multiscale representation for anti-aliasing neu-
ral radiance fields. In ICCV, 2021. 5
[2] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter
Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan.
Mip-nerf: A multiscale representation for anti-aliasing neu-
ral radiance fields. In Proceedings of the IEEE/CVF Interna-
tional Conference on Computer Vision (ICCV), pages 5855–
5864, 2021. 2, 3, 5, 1, 4, 6
[3] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P.
Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded
anti-aliased neural radiance fields.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 5470–5479, 2022. 2, 3, 5, 7,
8, 1
[4] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P
Srinivasan, and Peter Hedman.
Zip-nerf:
Anti-aliased
grid-based neural radiance fields.
In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 19697–19705, 2023. 3
[5] Robert A Brebin, Loren Carpenter, and Pat Hanrahan. Vol-
ume rendering. In Seminal graphics: pioneering efforts that
shaped the field, pages 363–372. 1998. 1
[6] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and
Hao Su. Tensorf: Tensorial radiance fields. In ECCV, 2022.
2, 6
[7] Franklin C Crow.
The aliasing problem in computer-
generated shaded images. Communications of the ACM, 20
(11):799–805, 1977. 1
[8] Leila De Floriani and Enrico Puppo. Hierarchical triangula-
tion for multiresolution surface description. ACM Transac-
tions On Graphics (TOG), 14(4):363–411, 1995. 2, 4
[9] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong
Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels:
Radiance fields without neural networks. In CVPR, 2022. 2,
6
[10] Wenbo Hu, Yuling Wang, Lin Ma, Bangbang Yang, Lin Gao,
Xiao Liu, and Yuewen Ma.
Tri-miprf: Tri-mip represen-
tation for efficient anti-aliasing neural radiance fields.
In
Proceedings of the IEEE/CVF International Conference on
Computer Vision (ICCV), pages 19774–19783, 2023. 3, 5, 6
[11] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4), 2023. 1, 2, 3, 5, 6, 8, 7
[12] Eric LaMar, Bernd Hamann, and Kenneth I Joy. Multireso-
lution techniques for interactive texture-based volume visu-
alization. IEEE, 1999. 2, 4
[13] Christoph Lassner and Michael Zollhofer.
Pulsar: Effi-
cient sphere-based neural rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 1440–1449, 2021. 2
[14] Marc Levoy.
Efficient ray tracing of volume data.
ACM
Transactions on Graphics (ToG), 9(3):245–261, 1990. 1
[15] Jiameng Li, Yue Shi, Jiezhang Cao, Bingbing Ni, Wenjun
Zhang, Kai Zhang, and Luc Van Gool. Mipmap-gs: Let gaus-
sians deform with scale-specific mipmap for anti-aliasing
rendering. In International Conference on 3D Vision 2025.
3, 5, 6, 8, 1, 7
[16] Zhihao Liang, Qi Zhang, Wenbo Hu, Lei Zhu, Ying Feng,
and Kui Jia.
Analytic-splatting:
Anti-aliased 3d gaus-
sian splatting via analytic integration. In Computer Vision
– ECCV 2024: 18th European Conference, Milan, Italy,
September 29–October 4, 2024, Proceedings, Part XVII,
page 281–297, Berlin, Heidelberg, 2024. Springer-Verlag. 3,
5, 6, 8, 1, 7
[17] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and
Christian Theobalt. Neural sparse voxel fields. 2020. 2
[18] David Luebke, Benjamin Watson, Jonathan D. Cohen, Mar-
tin Reddy, and Amitabh Varshney. Level of Detail for 3D
Graphics. Elsevier Science Inc., USA, 2002. 2, 4
[19] Nelson Max. Optical models for direct volume rendering.
TVCG, 1(2):99–108, 1995. 1
[20] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. In ECCV, 2020. 1, 2, 5, 6, 4, 7
[21] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexan-
der Keller. Instant neural graphics primitives with a multires-
olution hash encoding. TOG, 41(4):1–15, 2022. 2, 6
[22] Harry Nyquist. Certain topics in telegraph transmission the-
ory. Transactions of the American Institute of Electrical En-
gineers, 47(2):617–644, 2009. 2
[23] Sergey Prokudin,
Qianli Ma,
Maxime Raafat,
Julien
Valentin, and Siyu Tang. Dynamic point fields. In Proceed-
ings of the IEEE/CVF International Conference on Com-
puter Vision, pages 7964–7976, 2023. 2
[24] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu,
Zhangkai Ni, and Bo Dai. Octree-gs: Towards consistent
real-time rendering with lod-structured 3d gaussians. IEEE
Transactions on Pattern Analysis and Machine Intelligence,
2025. 3
[25] Claude E Shannon. Communication in the presence of noise.
Proceedings of the IRE, 37(1):10–21, 1949. 2
[26] Xiaowei Song, Jv Zheng, Shiran Yuan, Huan-ang Gao, Jing-
wei Zhao, Xiang He, Weihao Gu, and Hao Zhao.
Sa-
gs: Scale-adaptive gaussian splatting for training-free anti-
aliasing. arXiv preprint arXiv:2403.19615, 2024. 3, 5
[27] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel
grid optimization: Super-fast convergence for radiance fields
reconstruction. In CVPR, 2022. 2
[28] Weiwei Sun,
Eduard Trulls,
Yang-Che Tseng,
Sneha
Sambandam, Gopal Sharma, Andrea Tagliasacchi, and
Kwang Moo Yi.
Pointnerf++: a multi-scale, point-based
neural radiance field. In European Conference on Computer
Vision, pages 221–238. Springer, 2024. 2
[29] Chen Wang, Xian Wu, Yuan-Chen Guo, Song-Hai Zhang,
Yu-Wing Tai, and Shi-Min Hu. Nerf-sr: High-quality neural
radiance fields using supersampling. In Proceedings of the
30th ACM International Conference on Multimedia, pages
6445–6454, 2022. 4

<!-- page 10 -->
[30] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P
Simoncelli. Image quality assessment: from error visibility
to structural similarity. TIP, 13(4):600–612, 2004. 6
[31] Manfred Weiler, R¨udiger Westermann, Chuck Hansen, Kurt
Zimmermann, and Thomas Ertl. Level-of-detail volume ren-
dering via 3d textures.
In Proceedings of the 2000 IEEE
symposium on Volume visualization, pages 7–13, 2000. 2, 4
[32] Lance Williams. Pyramidal parametrics. In Proceedings of
the 10th Annual Conference on Computer Graphics and In-
teractive Techniques, page 1–11, New York, NY, USA, 1983.
Association for Computing Machinery. 2
[33] Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin
Shu, Kalyan Sunkavalli, and Ulrich Neumann.
Point-
nerf: Point-based neural radiance fields. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 5438–5448, 2022. 2
[34] Zhiwen Yan, Weng Fei Low, Yu Chen, and Gim Hee Lee.
Multi-scale 3d gaussian splatting for anti-aliased render-
ing. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition (CVPR), pages 20923–
20931, 2024. 3, 5, 6, 8, 1, 7
[35] Wang Yifan, Felice Serena, Shihao Wu, Cengiz ¨Oztireli, and
Olga Sorkine-Hornung. Differentiable surface splatting for
point-based geometry processing.
ACM Transactions On
Graphics (TOG), 38(6):1–14, 2019. 2
[36] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and
Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splat-
ting. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition (CVPR), pages 19447–
19456, 2024. 3, 4, 5, 6, 8, 1, 7
[37] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian
opacity fields: Efficient adaptive surface reconstruction in
unbounded scenes. ACM Trans. Graph., 43(6), 2024. 5
[38] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman,
and Oliver Wang. The unreasonable effectiveness of deep
features as a perceptual metric. In CVPR, 2018. 6
[39] Yufeng Zheng, Wang Yifan, Gordon Wetzstein, Michael J
Black, and Otmar Hilliges.
Pointavatar:
Deformable
point-based head avatars from videos.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 21057–21067, 2023. 2
[40] Yiyu Zhuang, Qi Zhang, Ying Feng, Hao Zhu, Yao Yao, Xi-
aoyu Li, Yan-Pei Cao, Ying Shan, and Xun Cao. Anti-aliased
neural implicit surfaces with encoding level of detail. New
York, NY, USA, 2023. Association for Computing Machin-
ery. 3
[41] Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and
Markus Gross. Ewa volume splatting. In Proceedings Visu-
alization, 2001. VIS’01., pages 29–538. IEEE, 2001. 2, 3,
5

<!-- page 11 -->
LOD-GS: Level of Detail-Sensitive 3D Gaussian Splatting for
Detail Conserved Anti-Aliasing
Supplementary Material
In this supplementary material, we provide a comprehen-
sive ablation study and its analysis in Section 7. We also
report the complete results of comparison experiments con-
ducted on the Multi-scale Blender Dataset [2], our newly
rendered Multi-level Blender Dataset, and the Mip-NeRF
360 Dataset [3] in Section 8.
7. Ablation
The capability of our method to achieve aliasing-free and
detail-conserved rendering relies on two components: the
level-of-detail sensitive filter (denoted as LOD filter here-
after) and the EWA filter. To assess the contribution of these
modules to overall performance of our method, we sepa-
rately remove the LOD and EWA filters and conduct model
training on synthetic and real-world dataset. Detailed anal-
yses of these two components are presented in Section 7.1
and Section 7.2.
7.1. Effectiveness of the LOD filter
The LOD filter is mainly responsible for disentangling in-
puts with different sampling rates. It helps alleviate the am-
biguity caused by the multi-scale input, which is the key to
learn a radiance field with fine details and anti-aliasing abil-
ity at the same time. As shown in Figure 8, scenes trained
without the LOD filter fail to capture the fine details of the
objects. For instance, the inner structure of the microphone
is unclear when photographed from a near distance. Other
details, such as the metal accents and rope net of the ship,
as well as the bumps on the chair, cannot be well recon-
structed without the LOD filter. Moreover, as illustrated in
Figure 9, using only the LOD filter can mitigate aliasing
to some degree compared to the vanilla 3DGS [11]. Ex-
perimental results in Table 6 and Table 7 demonstrate that
removing the LOD filter leads to a more significant perfor-
mance drop compared to removing the EWA filter. All these
results illustrate that the LOD filter in our method is crucial
for anti-aliasing and the conservation of fine details.
7.2. Effectiveness of the EWA filter
The EWA filter is utilized to enhance the model’s anti-
aliasing capability. Although our LOD filter can address the
aliasing problem to some extent, integrating the EWA filter
into our method can provide improved anti-aliasing perfor-
mance. As shown in Figure 8, the results rendered from a
far distance using the method that only utilizes the LOD fil-
ter face issues of aliasing and dilation artifacts. More specif-
ically, the result for mic is aliased, while the results for ship
and lego exhibit dilation artifacts. The quantitative results
in Table 6 and Table 7 illustrate that removing the EWA fil-
ter also leads to a performance drop. These observations
and results demonstrate that the EWA filter contributes to
the anti-aliasing capability of our method.
8. Additional Results
In this section, we report the detail of our comparison ex-
periment on each scene across different sampling rates. We
analyze the experimental results on the Blender Dataset in
Section 8.1 and the Mip-NeRF 360 Dataset in Section 8.2.
8.1. Blender Dataset
We perform comparison experiments on the Multi-Scale
Blender Dataset [2] and our newly rendered Multi-Level
Dataset. The Multi-Level Blender dataset does not contain
the drum scene because of the wrong specular reflection ef-
fect in the drums blender project. The experimental results
on the Multi-Scale Blender dataset are shown in Table 8.
Compared to existing powerful methods for radiance field
anti-aliasing, such as Mip-NeRF [2], Mip-Splatting [36],
and Analytic-Splatting [16], our method achieves state-of-
the-art performance in most scenes. We collected a new
dataset called the Multi-Level Blender dataset, which sim-
ulates changes in sampling rate using varying camera dis-
tances. The experimental results on this new dataset are
reported in Table 9. We compare our method with Gaus-
sian Splatting-based methods designed to handle multi-
scale inputs, including Mipmap-GS [15], MSGS [34], Mip-
Splatting [36], and Analytic-Splatting [16]. As shown in
Table 9, our method achieves the best performance in most
scenes. These experimental results demonstrate the state-
of-the-art performance of our method in handling inputs
with varying sampling rates.
8.2. Mip-NeRF 360 Dataset
We down-sample the full resolution images in Mip-NeRF
360 dataset in three scales (1/8, 1/16 and 1/32). All methods
are trained and tested on these three scales. The experi-
mental results are displayed in the Table 10. As shown, our
proposed LOD-GS achieves the best or nearly the best per-
formance across all scenes. This illustrates that our method
is also applicable to wild and unbounded real-world data.

<!-- page 12 -->
3DGS
w/o LOD
w/o EWA
Full Model
GT
far
near
far
near
far
near
far
near
Figure 8. Qualitative Results of the Ablation Study on the Blender Dataset [2, 20]. We render the objects from both far and near
distances to test the anti-aliasing and detail conservation abilities.

<!-- page 13 -->
GT
3DGS
w/o LOD
w/o EWA
Full Model
1/8
1/32
1/8
1/32
1/8
1/32
1/8
1/32
Figure 9. Qualitative Results of the Ablation Study on the Mip-NeRF 360 Dataset [3]. All methods are trained and tested on multiscale
inputs (1/8, 1/16, 1/32). We present rendering results from different methods at 1/8 and 1/32 scales to evaluate detail conservation and anti-
aliasing performance.

<!-- page 14 -->
PSNR
chair
drums
ficus
hotdog
lego
materials
mic
ship
Average
Full Model Full Res.
34.99
25.95
35.06
37.28
34.76
29.66
34.90
30.61
32.90
Full Model 1/2
38.49
27.16
35.94
39.44
36.43
31.21
37.78
32.59
34.88
Full Model 1/4
40.39
28.65
36.38
40.99
37.23
33.22
40.23
34.31
36.43
Full Model 1/8
41.37
30.28
36.26
41.77
36.84
35.10
40.84
35.70
37.27
w/o LOD Full Res.
33.23
25.73
34.08
36.54
33.29
29.04
33.44
29.70
31.88
w/o LOD 1/2
37.04
27.02
35.20
38.90
35.35
30.79
36.46
31.92
34.09
w/o LOD 1/4
39.43
28.51
35.95
40.56
36.45
33.05
39.07
33.89
35.86
w/o LOD 1/8
38.68
29.78
35.53
40.24
34.77
33.79
38.38
34.69
35.73
w/o EWA Full Res.
34.59
25.88
34.51
37.15
34.44
29.62
34.17
30.66
32.63
w/o EWA 1/2
37.77
26.98
34.49
39.29
35.96
31.14
36.95
32.58
34.39
w/o EWA 1/4
39.31
28.27
33.89
40.81
36.45
32.88
39.55
34.11
35.66
w/o EWA 1/8
38.93
29.24
33.67
40.79
35.21
33.96
39.46
34.80
35.76
SSIM
chair
drums
ficus
hotdog
lego
materials
mic
ship
Average
Full Model Full Res.
0.983
0.950
0.986
0.983
0.979
0.958
0.990
0.901
0.966
Full Model 1/2
0.992
0.959
0.992
0.990
0.988
0.974
0.993
0.927
0.977
Full Model 1/4
0.995
0.968
0.994
0.993
0.992
0.987
0.995
0.952
0.985
Full Model 1/8
0.997
0.978
0.994
0.996
0.994
0.994
0.997
0.969
0.990
w/o LOD Full Res.
0.973
0.946
0.984
0.981
0.973
0.954
0.986
0.894
0.961
w/o LOD 1/2
0.989
0.958
0.991
0.989
0.986
0.973
0.991
0.924
0.975
w/o LOD 1/4
0.994
0.967
0.994
0.993
0.992
0.986
0.995
0.949
0.984
w/o LOD 1/8
0.995
0.976
0.993
0.995
0.992
0.993
0.997
0.968
0.989
w/o EWA Full Res.
0.981
0.950
0.986
0.983
0.978
0.959
0.988
0.903
0.966
w/o EWA 1/2
0.992
0.959
0.990
0.989
0.988
0.974
0.992
0.928
0.977
w/o EWA 1/4
0.995
0.966
0.990
0.993
0.992
0.986
0.995
0.950
0.983
w/o EWA 1/8
0.995
0.974
0.989
0.995
0.992
0.992
0.996
0.966
0.987
LPIPS
chair
drums
ficus
hotdog
lego
materials
mic
ship
Average
Full Model Full Res.
0.018
0.044
0.014
0.025
0.023
0.040
0.008
0.112
0.035
Full Model 1/2
0.007
0.030
0.007
0.011
0.009
0.017
0.004
0.066
0.019
Full Model 1/4
0.005
0.026
0.005
0.006
0.006
0.009
0.004
0.035
0.012
Full Model 1/8
0.003
0.021
0.005
0.004
0.006
0.005
0.004
0.019
0.008
w/o LOD Full Res.
0.029
0.050
0.018
0.028
0.031
0.047
0.013
0.122
0.042
w/o LOD 1/2
0.011
0.032
0.009
0.012
0.011
0.019
0.006
0.069
0.021
w/o LOD 1/4
0.006
0.027
0.006
0.006
0.007
0.010
0.004
0.037
0.013
w/o LOD 1/8
0.006
0.023
0.006
0.004
0.008
0.007
0.006
0.021
0.010
w/o EWA Full Res.
0.022
0.045
0.014
0.027
0.025
0.040
0.010
0.119
0.038
w/o EWA 1/2
0.009
0.031
0.009
0.012
0.010
0.017
0.005
0.067
0.020
w/o EWA 1/4
0.005
0.027
0.009
0.006
0.007
0.010
0.004
0.036
0.013
w/o EWA 1/8
0.004
0.024
0.009
0.004
0.008
0.007
0.005
0.021
0.010
Table 6. Quantitative Results of the Ablation Study on the Blender Dataset [2, 20]. We present the experimental results of our full
model alongside methods that exclude the LOD module or EWA filter across different scenes and resolutions. The results demonstrate that
both the LOD and EWA filters significantly contribute to the overall performance of our method.

<!-- page 15 -->
PSNR
bicycle
bonsai
counter
flowers
garden
kitchen
room
stump
treehill
Average
Full Model 1/8
27.08
32.89
30.19
23.76
29.05
32.81
33.03
27.67
24.38
28.99
Full Model 1/16
29.37
34.60
31.63
26.77
31.64
34.37
33.90
29.90
26.19
30.93
Full Model 1/32
30.95
35.52
32.98
29.42
33.28
35.50
34.61
32.07
28.02
32.48
w/o LOD 1/8
26.95
32.02
29.78
23.56
28.61
31.99
32.60
27.56
24.18
28.59
w/o LOD 1/16
29.17
33.77
31.30
26.43
31.14
33.82
33.64
29.69
26.00
30.55
w/o LOD 1/32
30.23
33.86
32.13
28.77
32.30
33.90
33.72
31.41
27.57
31.54
w/o EWA 1/8
27.09
32.56
29.93
23.66
28.96
32.27
32.56
27.80
24.46
28.81
w/o EWA 1/16
29.29
34.25
31.26
26.54
31.51
33.76
33.42
29.74
26.15
30.66
w/o EWA 1/32
30.41
34.44
31.87
28.50
32.75
34.08
33.76
31.16
27.42
31.60
SSIM
bicycle
bonsai
counter
flowers
garden
kitchen
room
stump
treehill
Average
Full Model 1/8
0.833
0.963
0.934
0.719
0.901
0.964
0.962
0.825
0.729
0.870
Full Model 1/16
0.921
0.978
0.960
0.841
0.957
0.981
0.975
0.900
0.845
0.929
Full Model 1/32
0.954
0.986
0.975
0.909
0.974
0.989
0.982
0.942
0.913
0.958
w/o LOD 1/8
0.823
0.954
0.927
0.711
0.889
0.954
0.957
0.817
0.721
0.862
w/o LOD 1/16
0.917
0.974
0.958
0.839
0.952
0.974
0.973
0.897
0.841
0.925
w/o LOD 1/32
0.947
0.980
0.972
0.907
0.967
0.982
0.979
0.936
0.907
0.953
w/o EWA 1/8
0.832
0.962
0.932
0.713
0.900
0.962
0.959
0.826
0.733
0.869
w/o EWA 1/16
0.919
0.978
0.958
0.836
0.956
0.980
0.972
0.897
0.844
0.927
w/o EWA 1/32
0.945
0.983
0.970
0.897
0.971
0.985
0.978
0.925
0.902
0.951
LPIPS
bicycle
bonsai
counter
flowers
garden
kitchen
room
stump
treehill
Average
Full Model 1/8
0.178
0.052
0.074
0.257
0.092
0.039
0.058
0.169
0.276
0.133
Full Model 1/16
0.067
0.021
0.038
0.141
0.030
0.017
0.027
0.085
0.157
0.065
Full Model 1/32
0.036
0.012
0.021
0.083
0.018
0.010
0.018
0.054
0.090
0.038
w/o LOD 1/8
0.183
0.061
0.082
0.261
0.098
0.049
0.064
0.176
0.283
0.140
w/o LOD 1/16
0.071
0.027
0.041
0.141
0.035
0.029
0.030
0.089
0.161
0.069
w/o LOD 1/32
0.042
0.018
0.027
0.085
0.025
0.021
0.022
0.060
0.094
0.044
w/o EWA 1/8
0.184
0.054
0.078
0.268
0.097
0.042
0.061
0.175
0.281
0.138
w/o EWA 1/16
0.068
0.021
0.040
0.144
0.031
0.019
0.029
0.089
0.156
0.066
w/o EWA 1/32
0.042
0.014
0.026
0.092
0.021
0.014
0.021
0.064
0.091
0.043
Table 7. Quantitative Results of the Ablation Study on the Mip-NeRF 360 Dataset [3]. All methods are trained and tested on multiscale
inputs. We present the experimental results of our full model alongside methods that exclude either the LOD module or the EWA filter
across different scenes and resolutions.

<!-- page 16 -->
PSNR
chair
drums
ficus
hotdog
lego
materials
mic
ship
Average
NeRF [20]
33.39
25.87
30.37
35.64
31.65
30.18
32.60
30.09
31.23
Mip-NeRF [2]
37.14
27.02
33.19
39.31
35.74
32.56
38.04
33.08
34.51
Plenoxels [9]
32.79
25.25
30.28
34.65
31.26
28.33
31.53
28.59
30.34
TensoRF [6]
32.47
25.37
31.16
34.96
31.73
28.53
31.48
29.08
30.60
Instant-ngp [21]
32.95
26.43
30.41
35.87
31.83
29.31
32.58
30.23
31.20
Tri-MipRF [10]*
37.67
27.35
33.57
38.78
35.72
31.42
37.63
32.74
34.36
3DGS [11]
32.73
25.30
29.00
35.03
29.44
27.13
31.17
28.33
29.77
Mipmap-GS [15]
32.72
25.30
29.01
35.01
29.45
27.14
31.17
28.43
29.78
MSGS [34]
27.00
21.16
25.97
28.80
25.35
23.14
24.46
23.16
24.88
Mip-Splatting [36]
37.48
27.74
34.71
39.15
35.07
31.88
37.68
32.80
34.56
Analytic-Splatting [16]
38.26
27.98
36.11
39.47
35.75
31.74
37.78
33.13
35.03
LOD-GS (ours)
38.86
28.01
35.90
39.86
36.20
32.33
38.41
33.38
35.37
SSIM
chair
drums
ficus
hotdog
lego
materials
mic
ship
Average
NeRF [20]
0.971
0.932
0.971
0.979
0.965
0.967
0.980
0.900
0.958
Mip-NeRF [2]
0.988
0.945
0.984
0.988
0.984
0.977
0.993
0.922
0.973
Plenoxels [9]
0.968
0.929
0.972
0.976
0.964
0.959
0.979
0.892
0.955
TensoRF [6, 6]
0.967
0.930
0.974
0.977
0.967
0.957
0.978
0.895
0.956
Instant-ngp [21]
0.971
0.940
0.973
0.979
0.966
0.959
0.981
0.904
0.959
Tri-MipRF [10]*
0.990
0.951
0.985
0.988
0.986
0.969
0.992
0.929
0.974
3DGS [11]
0.976
0.941
0.968
0.982
0.964
0.956
0.979
0.910
0.960
Mipmap-GS [15]
0.976
0.941
0.968
0.982
0.964
0.956
0.979
0.911
0.960
MSGS [34]
0.915
0.849
0.920
0.929
0.884
0.883
0.910
0.828
0.890
Mip-Splatting [36]
0.991
0.963
0.990
0.990
0.987
0.978
0.994
0.936
0.979
Analytic-Splatting [16]
0.992
0.964
0.992
0.991
0.988
0.977
0.994
0.936
0.979
LOD-GS (ours)
0.992
0.964
0.992
0.991
0.988
0.978
0.994
0.938
0.980
LPIPS
chair
drums
ficus
hotdog
lego
materials
mic
ship
Average
NeRF [20]
0.028
0.059
0.026
0.024
0.035
0.033
0.025
0.085
0.044
Mip-NeRF [2]
0.011
0.044
0.014
0.012
0.013
0.019
0.007
0.062
0.026
Plenoxels [9]
0.040
0.070
0.032
0.037
0.038
0.055
0.036
0.104
0.051
TensoRF [6]
0.042
0.075
0.032
0.035
0.036
0.063
0.040
0.112
0.054
Instant-ngp [21]
0.035
0.066
0.029
0.028
0.040
0.051
0.032
0.095
0.047
Tri-MipRF [10]*
0.011
0.046
0.016
0.014
0.013
0.033
0.008
0.069
0.026
3DGS [11]
0.025
0.056
0.030
0.022
0.038
0.040
0.023
0.086
0.040
Mipmap-GS [15]
0.025
0.055
0.030
0.022
0.038
0.040
0.023
0.086
0.040
MSGS [34]
0.046
0.090
0.056
0.037
0.065
0.057
0.048
0.113
0.064
Mip-Splatting [36]
0.010
0.031
0.009
0.011
0.012
0.018
0.005
0.059
0.019
Analytic-Splatting [16]
0.008
0.029
0.007
0.011
0.011
0.018
0.005
0.058
0.018
LOD-GS (ours)
0.007
0.028
0.008
0.011
0.011
0.018
0.005
0.058
0.018
Table 8. Multi-scale Training and Multi-scale Testing on the the Blender dataset [20]. For each scene, we report the arithmetic mean
of each metric averaged over the four scales used in the dataset (full resolution, 1/2, 1/4, and 1/8 downsampled scales).

<!-- page 17 -->
PSNR
chair
ficus
hotdog
lego
materials
mic
ship
Average
3DGS [11]
33.71
32.28
33.10
33.14
33.36
30.64
29.86
32.30
Mimpmap-GS [15]
32.80
30.86
32.55
31.20
31.72
29.21
28.33
30.95
MSGS [34]
32.78
29.86
32.56
31.19
31.76
29.33
28.37
30.84
Mip-Splatting[36]
35.23
33.42
34.09
35.06
34.93
31.00
31.27
33.57
Analytic-Splatting[16]
35.63
33.52
34.26
35.00
34.71
30.72
31.24
33.58
LOD-GS(ours)
35.90
33.94
34.39
35.34
35.09
31.58
31.69
33.99
SSIM
chair
ficus
hotdog
lego
materials
mic
ship
Average
3DGS [11]
0.971
0.975
0.962
0.958
0.977
0.947
0.910
0.957
Mimpmap-GS [15]
0.967
0.971
0.960
0.953
0.974
0.941
0.902
0.952
MSGS [34]
0.967
0.969
0.960
0.953
0.974
0.942
0.902
0.952
Mip-Splatting[36]
0.972
0.976
0.962
0.960
0.979
0.943
0.909
0.957
Analytic-Splatting[16]
0.973
0.977
0.962
0.960
0.977
0.939
0.908
0.957
LOD-GS(ours)
0.973
0.977
0.963
0.961
0.978
0.947
0.910
0.958
LPIPS
chair
ficus
hotdog
lego
materials
mic
ship
Average
3DGS [11]
0.026
0.028
0.111
0.063
0.041
0.056
0.108
0.062
Mimpmap-GS [15]
0.028
0.030
0.113
0.068
0.043
0.061
0.115
0.066
MSGS [34]
0.028
0.032
0.113
0.068
0.043
0.060
0.115
0.066
Mip-Splatting[36]
0.026
0.027
0.114
0.062
0.039
0.058
0.109
0.062
Analytic-Splatting[16]
0.024
0.026
0.111
0.061
0.041
0.062
0.111
0.062
LOD-GS(ours)
0.025
0.025
0.109
0.059
0.040
0.055
0.108
0.060
Table 9. Multi-Level Training and Multi-Level Testing on our extended Blender dataset [20]. For each scene, we report the arithmetic
mean of each metric averaged over the 3 Levels used in the dataset (near, middle and far).

<!-- page 18 -->
PSNR
bicycle
bonsai
counter
flowers
garden
kitchen
room
stump
treehill
Average
3DGS [11]
27.74
31.94
29.95
25.12
30.05
31.51
32.23
28.57
25.62
29.19
Mipmap-GS [15]
26.99
30.25
28.75
24.28
29.45
30.84
31.19
27.95
25.32
28.33
MSGS [34]
26.93
30.32
28.75
24.31
29.42
30.67
31.30
27.97
25.47
28.35
Mip-Splatting[36]
28.65
32.69
30.50
25.86
29.93
32.55
33.06
28.96
26.48
29.86
Analytic-Splatting[16]
28.76
33.52
31.11
26.37
30.81
33.78
33.41
29.56
26.01
30.37
LOD-GS(ours)
29.14
34.34
31.60
26.65
31.32
34.23
33.85
29.88
26.19
30.80
SSIM
bicycle
bonsai
counter
flowers
garden
kitchen
room
stump
treehill
Average
3DGS [11]
0.877
0.964
0.943
0.795
0.928
0.964
0.965
0.862
0.811
0.901
Mipmap-GS [15]
0.860
0.952
0.930
0.770
0.921
0.961
0.955
0.842
0.803
0.888
MSGS [34]
0.860
0.953
0.930
0.769
0.921
0.960
0.957
0.842
0.805
0.888
Mip-Splatting[36]
0.899
0.971
0.951
0.816
0.938
0.966
0.970
0.874
0.831
0.913
Analytic-Splatting[16]
0.897
0.972
0.954
0.821
0.940
0.975
0.970
0.884
0.825
0.915
LOD-GS(ours)
0.902
0.976
0.956
0.824
0.944
0.979
0.973
0.888
0.829
0.919
LPIPS
bicycle
bonsai
counter
flowers
garden
kitchen
room
stump
treehill
Average
3DGS [11]
0.115
0.043
0.060
0.183
0.062
0.037
0.045
0.130
0.192
0.096
Mipmap-GS [15]
0.131
0.053
0.072
0.204
0.069
0.039
0.056
0.150
0.199
0.108
MSGS [34]
0.131
0.052
0.072
0.204
0.069
0.039
0.054
0.150
0.200
0.108
Mip-Splatting[36]
0.090
0.030
0.049
0.146
0.053
0.037
0.036
0.119
0.156
0.080
Analytic-Splatting[16]
0.093
0.032
0.046
0.157
0.048
0.027
0.038
0.103
0.170
0.079
LOD-GS(ours)
0.093
0.028
0.044
0.160
0.047
0.021
0.034
0.103
0.173
0.078
Table 10. Multi-Scale Training and Multi-Scale Testing on the the Mip-NeRF 360 dataset [3]. For each scene, we report the arithmetic
mean of each metric averaged over the 3 scales used in the dataset (1/8, 1/16, and 1/32 downsampled scales).
