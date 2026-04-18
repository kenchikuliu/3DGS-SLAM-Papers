<!-- page 1 -->
POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality
Joey Wilson
University of Michigan
wilsoniv@umich.edu
Marcelino Almeida
Amazon Lab 126
mmalmeid@amazon.com
Sachit Mahajan
Amazon Lab 126
msachit@amazon.com
Martin Labrie
Amazon Lab 126
labrieml@amazon.com
Maani Ghaffari
University of Michigan
maanigj@umich.edu
Omid Ghasemalizadeh
Amazon Lab 126
ghasemal@amazon.com
Min Sun
Amazon Lab 126
aliensunmin@gmail.com
Cheng-Hao Kuo
Amazon Lab 126
chkuo@amazon.com
Arnab Sen
Amazon Lab 126
senarnie@amazon.com
Abstract
In this paper, we present a novel algorithm for quantify-
ing uncertainty and information gained within 3D Gaussian
Splatting (3D-GS) through P-Optimality. While 3D-GS has
proven to be a useful world model with high-quality raster-
izations, it does not natively quantify uncertainty or infor-
mation, posing a challenge for real-world applications such
as 3D-GS SLAM. We propose to quantify information gain
in 3D-GS by reformulating the problem through the lens
of optimal experimental design, which is a classical solu-
tion widely used in literature. By restructuring information
quantification of 3D-GS through optimal experimental de-
sign, we arrive at multiple solutions, of which T-Optimality
and D-Optimality perform the best quantitatively and qual-
itatively as measured on two popular datasets. Addition-
ally, we propose a block diagonal covariance approxima-
tion which provides a measure of correlation at the expense
of a greater computation cost.
1. Introduction
In order to operate in a novel environment, robots must be
capable of quantifying uncertainty in their surroundings due
to occlusions and unobserved regions, as well as quanti-
fying information gained by exploring new areas. While
many prior methods for map representations were built on
a metric space with natural uncertainty quantification, such
representations result in discretization errors of the environ-
ment [39]. Recently, 3D-Gaussian Splatting (3D-GS) was
proposed as a method for high-quality novel-view synthesis
[15], which captivated the attention of the robotics commu-
nity due to its explicit world model representation, with the
(a) Uniform
(b) FisherRF
(c) Ours
(d) Ground Truth
Figure 1. We propose a novel method for calculating information
gain from image in 3D Gaussian Splatting. In the above images,
each method is provided a set of one hundred candidate views on
scenes from the Blender dataset, and selects ten views to train a
3D-GS model on. Compared to state-of-the-art, our method more
accurately estimate the information value of images to train a 3D-
GS model. In this figure we demonstrate block D-Optimality, how-
ever our derivation also provides multiple solutions discussed later.
potential to function as a map [7, 24, 38, 43].
Given a set of views of a scene, 3D-GS learns to ren-
der novel views from any angle through gradient descent.
3D ellipsoids are iteratively fine-tuned, which can then be
rasterized to a 2D image at novel views. Several works
have shown that the 3D ellipsoids can be extended to in-
clude features beyond just color, such as the category of ob-
jects the ellipsoid represents, resulting in an improved level
of semantic scene understanding [31, 42]. However, 3D-GS
does not natively quantify uncertainty, which leads to issues
when determining whether an image has been seen before,
arXiv:2503.07819v2  [cs.CV]  25 Mar 2025

<!-- page 2 -->
or when quantifying the information gained by perceiving a
new view. While several works have studied this problem
for neural radiance fields (NeRF) [9, 26, 27], the explicit
representation of 3D-GS presents a unique challenge.
Since the conception of 3D-GS, several works have
sought to quantify uncertainty directly upon a trained 3D-
GS model [10, 14, 19]. Of these works, one solution which
has appeared effective is relating the per-pixel gradient of
the 3D-GS parameters to the information or contribution of
the parameters. Notably, FisherRF [14] derived a solution
for information quantification of views in 3D-GS through a
diagonal approximation of Fisher Information [17]. Fish-
erRF has since been applied to mobile manipulation [34]
and active perception [22], however it does not leverage
prior literature in effective functions for active perception
[16, 33] or consider any correlation between parameters.
Classically, the problem of quantifying information gain
from views is known as active perception in robotics litera-
ture, and has been studied extensive in probabilistic robotics
applications [8, 23]. More generally, active perception can
be solved through application of optimal experimental de-
sign [20], which defines solutions for identifying the most
informative design parameters for an experiment in order to
learn about a set of unknown parameters [11, 12, 18, 21].
One particular solution to experimental design is the use of
P-Optimality, which specifies a class of solutions depending
on the covariance matrix and the choice of P. P-Optimality
has been successfully applied in robotics to keyframe selec-
tion, optimal loop closure, and active perception, however
has not been explored in 3D-GS [3, 29, 30].
Building upon the recent work of FisherRF, which shows
that a diagonal approximation of the Hessian matrix can
be effective when calculating information gain in 3D-GS,
we derive a general solution for the covariance matrix in
3D-GS and apply optimal experimental design techniques.
Since the covariance matrix is too large to store in mem-
ory, we propose a simple diagonal approximation following
FisherRF [14], as well as a block diagonal approximation
which captures cross-correlation of parameters within the
same ellipsoid [10]. From our general derivation, we con-
struct and compare different p-optimality solutions, and find
that D-Optimality and T-Optimality lead to significant im-
provements. We also find that our block diagonal approxi-
mation improves information gain quantification compared
to the baselines.
To summarize, our contributions are:
1. Derive a general solution for the covariance matrix of
3D-GS, which allows application of optimal experimen-
tal design techniques.
2. Propose several novel methods for quantifying the infor-
mation value of images in 3D-GS.
3. Block-diagonal approximation of 3D-GS covariance ma-
trix which captures correlation between parameters of
the same ellipsoid.
4. Quantitative and qualitative comparison of optimal ex-
perimental design solutions on 3D-GS to each-other, and
to current state-of-the-art solutions.
2. Related Work
In this section, we explore literature related to 3D Gaussian
Splatting (3D-GS) and optimal experimental design for ac-
tive perception. In this paper, we propose to leverage the
theory of p-optimality from optimal experimental design lit-
erature to quantify information gain within 3D-GS.
2.1. 3D Gaussian Splatting
3D Gaussian Splatting (3D-GS) is a new method for novel
view synthesis which models scenes through 3D ellipsoids
[15], different from previous methods which model scenes
implicitly [26, 35]. Each scene is modeled through thou-
sands or millions of 3D ellipsoids where each ellipsoid con-
tain an opacity and color modeled by spherical harmonics
to capture lighting effects. 3D ellipsoids are trained through
gradient descent, and at inference time are rasterized to cre-
ate 2D images at candidate views through a process known
as “splatting” [7].
Due to the explicit representation of scenes as 3D el-
lipsoids, 3D-GS has attracted a significant amount of at-
tention from the computer vision and robotics research
communities [24, 40, 43].
3D-GS has the potential to
substitute as a more expressive world model representa-
tion, and many works have explored adding additional fea-
tures such as from vision-language (VL) networks to cre-
ate higher levels of scene understanding [31, 42]. How-
ever, despite the name, 3D-GS does not provide a measure
of uncertainty which limits applications in safety-critical
or resource-constrained environments.
Recently, several
works have investigated uncertainty quantification and have
found a promising research direction of relating Fisher In-
formation to the explicit 3D-GS parameters [10, 14]. In
particular, FisherRF developed a formulation for calculat-
ing information gain which treats the 3D-GS model as a
black box, not requiring any additional training. However,
FisherRF does not leverage any inter-parameter correlation
when calculating the information of values to images, and
does not utilize the rich literature of optimal experimental
design and active perception which derive solutions for cal-
culating information gain.
2.2. Active Perception
Active perception is a well-studied problem in robotics lit-
erature which seeks to identify the optimal path to improve
a map. Since capturing new views in the real world requires
robot traversal, a significant amount of research has focused
on optimal solutions to determining which view-points are
most valuable. One successful approach is the application

<!-- page 3 -->
of P-optimality [16], which defines a class of optimal ex-
perimental design solutions based on functionals of the co-
variance matrix which vary with the choice of an integer
p.
P-Optimality (P-Opt.)
has been widely and success-
fully applied to SLAM prior to the conception of 3D-GS
in keyframe selection, optimal loop closure, and active per-
ception [3, 29, 30].
Early research in optimal decision-taking for Simultane-
ous Localization and Mapping problems used T-optimality
due to efficient computation as the trace of the covariance
matrix [25, 33], eliminating the need to compute the eigen-
values. On the other hand, recent research has focused on
D-optimality as a reliable metric for Optimal Experimental
Design [4, 28–30], especially in active mapping [3]. The re-
cent success of D-optimality can be explained by its mono-
tonicity property in active mapping scenarios [32], which
guarantees that uncertainty increases monotonically as a
robot moves through the scene. Additionally, from an infor-
mation theoretic perspective, differential entropy of a mul-
tivariate Gaussian is proportional to the determinant of the
covariance matrix, which is captured by D-optimality [5].
In this work, we propose to expand upon the formulation
from FisherRF to incorporate parameter correlation and de-
velop a more general solution which allows for application
of classical optimal experimental design techniques.
3. Method
In this section, we introduce our method for quantifying
uncertainty and information gain in 3D-Gaussian Splatting.
First, we introduce preliminaries on the 3D Gaussian Splat-
ting representation. Next, we describe our approximation
of the covariance matrix for each ellipsoid, which provides
a measure of uncertainty on the parameters. Finally, we de-
tail our method for efficiently calculating the informational
value of a candidate image.
3.1. Preliminaries: Gaussian Splatting
3D Gaussian splatting represents a scene through volumet-
ric rendering of optimized 3D ellipsoids. The geometry of
each 3D ellipsoid is parameterized by a center µ, scale S,
and rotation R, while the color contribution of each ellip-
soid is defined by opacity α and color c. Together, the rota-
tion and scale define the shape of the 3D ellipsoid:
  \Sig ma  = R S S^T R^T. 
(1)
To render images, 3D ellipsoids are first splatted into 2D
projections from a provided viewpoint, resulting in a 2D
shape Σ′ and location µ′. Next, the contribution α′
n of each
2D ellipsoid n to pixel x′ is calculated through a kernel as:
  
\ a lp h a _
n
' 
= \a l ph
a _ n \t
i mes  \t
ex
t
 {exp}\left ( - \frac {1}{2} (x' - \mu _n')^T {\Sigma '}_n^{-1} (x' - \mu _n') \right ). 
(2)
Finally, 2D ellipsoids are blended into pixels through a
process known as alpha compositing, which computes the
color of each pixel from a depthwise sorted list of Gaussians
N:
  
C
 
= \
sum 
_
{n=
1
}^{
\m a th
cal {N}} c_n \alpha _n' \prod _{j=1}^{n-1} (1 - \alpha _j'). 
(3)
Parameters are optimized by comparing the rendered
image from a viewpoint with the ground truth image and
performing gradient descent over a weighted loss function
[7, 15, 37]:
  \m a thca l  {L} = (1 - \lambda )\mathcal {L}_1 + \lambda \mathcal {L}_{\text {D-SSIM}}, 
(4)
where λ is a weighting function on the L1 and structural
similarity loss. Once the 3D-GS model is fitted to the train-
ing data, it can render views from any perspective, however
it lacks any information on when the rendering may fail. In
order to quantify the amount of information the fitted Gaus-
sian Splatting model has on each parameter, we note that
the L1 loss is proportional to the partial derivative of the
rendered pixel’s color with respect to the parameters of the
3D-GS model, summed over all pixels in the training set.
Based on this insight, we construct our covariance matrix to
capture this information.
3.2. Information Gain through Optimal Experi-
mental Design
In order to quantify information gained through adding an
image, we first require a measure of uncertainty, which is
derived in this section. Approaching the problem from a
maximum likelihood perspective, the maximum likelihood
formulation aims to determine the solution variable θ that
minimizes the pixel error e = c −h(θ) across all pixels
in all frames, where h(·) is the 3D-GS model described in
Section 3.1. Assuming that all measured pixel errors are
normal zero mean, independent and identically distributed
(IID), i.e, e ∼N(0, σe · I), then the maximum likelihood
formulation seeks an optimal solution variable θ ∈Rl×l
that maximizes the likelihood function:
 p(\bm  {c}
 
|  
\
bm  
{\t
h
eta } ) = \exp \begin {pmatrix} - \frac {1}{2}\frac {\bm {e}^T \bm {e}}{\bm {\sigma }_e^2} \end {pmatrix}.
(5)
Due to the monotonicity of the log function, maximiz-
ing the function above is equivalent to minimizing the log-
likelihood function:
  \
l a bel {eq:neg_log_likelihood_function} - \sigma _e^2 \cdot \text {log} \, p(\bm {c} | \bm {\theta } ) = \frac {1}{2} \bm {e}^T \bm {e} g
(6)
Assuming that we have an estimate of the solution vari-
ables θ∗, then we can expand the system’s model in the
vicinity of θ∗using Taylor expansion as: h(θ) ≈h(θ∗) +

<!-- page 4 -->
Figure 2. The Hessian matrix captures the information content of
each parameter in the trained 3D-GS model, approximated as per-
pixel gradients over the training set of images. Since a trained 3D-
GS model may contain millions of parameters, we approximate
the Hessian matrix through a block or main diagonal.
J∆θ, where ∆θ ≜θ −θ∗and J ≜
∂h
∂θ |θ=θ∗. We can
rewrite the residual function as:
  \ b m {e}  &\a
p pr o x \bm {c} - \bm {h}(\bm {\theta }_*) - \bm {J} \Delta \bm {\theta } \nonumber \\ &= \bm {e}_* - \bm {J} \Delta \bm {\theta },
(7)
where the optimal residual is defined as e∗≜c −h(θ∗).
Substituting this in Eq. 6, we have that:
 
 \l a b e
l {
e q: n eg
_ log _ l
ikel ih ood_function_expansion} \frac {1}{2} \bm {e}^T \bm {e} \approx \frac {1}{2} \bm {e}_{*}^T \bm {e}_{*} - \bm {e}_{*}^T \bm {J} \Delta \bm {\theta } + \frac {1}{2} \Delta \bm {\theta }^T \bm {J}^T \bm {J} \Delta \bm {\theta }.
(8)
In order to satisfy the first order conditions for optimality
in Eq. 8, it is necessary for its first order partial derivative
w.r.t. the solution variables to be zero [1]. This leads to the
Gauss-Newton iterative optimization equation where ∆θ is
updated as:
 \ D
e
lt a
 \b m {\theta } = \begin {pmatrix} \bm {J}^T \bm {J} \end {pmatrix}^{-1} \bm {J}^T \bm {e}_*.
(9)
Assuming that the optimal solution θ∗is unbiased,
then it follows that the expected residual E[e∗] = 0 and
E[e∗eT
∗] = σe · I, leading to E[∆θ] = 0 and:
  \math b b
 
{E }
[\D el ta \bm
 {\
t
he t
a }
 \D
e
l
ta  
\bm  { \t h
e
ta  
}^T
]  &
=
 
\b e
gin  {pmatrix} \bm {J}^T \bm {J} \end {pmatrix}^{-1} \bm {J}^T \mathbb {E}[\bm {e}_* \bm {e}_*^T] \bm {J} \begin {pmatrix} \bm {J}^T \bm {J} \end {pmatrix}^{-1} \nonumber \\ &= \sigma _e^2 \begin {pmatrix} \bm {J}^T \bm {J} \end {pmatrix}^{-1} \bm {J}^T \bm {J}^T \bm {J} \begin {pmatrix} \bm {J}^T \bm {J} \end {pmatrix}^{-1} \nonumber \\ &= \sigma _e^2 \begin {pmatrix} \bm {J}^T \bm {J} \end {pmatrix}^{-1}.
(10)
Without loss of generality, this work assumes1 σe = 1.
Defining H ≜JT J ∈Rl×l, the matrix H is known as
an approximation of the Hessian, or the information matrix
1Given that we’ve assumed that the pixel error is zero-mean IID, the
value of σe does not matter for this application.
Figure 3. Optimal experimental design defines functionals of the
eigenvalues of the covariance matrices, each with geometric intu-
itions. D-Optimality approximates the volume of the covariance
matrix, as shown in this figure.
for this nonlinear optimization problem. Therefore, the co-
variance matrix associated with the solution variables θ is
given by the inverse of the Hessian (information) matrix.
3.2.1. Uncertainty Decrease due to an Added Image
In this paper, we assume that we have a set of n im-
ages to choose from (along with their respective original
poses) to determine which of the images will lead to max-
imal uncertainty reduction among all n candidates. There-
fore, our Next-Best-View formulation attempts at maxi-
mally decreasing uncertainty of the covariance matrix Σi
that is obtained by adding the i-th image to the model,
i ∈{1, · · · , n}. Note that the contents of the i-th image are
not necessary for our formulation, only the pose at which we
plan to take the i-th image from. This is an important fea-
ture of our solution, as it attempts to evaluate the amount of
uncertainty reduction (or information increase) that can be
achieved by taking an image from a new perspective with-
out actually having the image available.
In this section, we assume that we already have an initial
guess of the map θ∗and that its associated prior Jacobian
J−and Hessian H−= JT
−J−have been computed. As
we add one new prospective candidate image i taken from
a pose pi, it is possible to compute the Jacobian associ-
ated with the new image using prior map parameters θ∗and
the image’s pose pi as Ji = ∂h
∂θ |θ∗,pi. Defining Hi as the
Hessian of the problem as we add the i-th image, it can be
computed as:
 \ b m { H}
_ i = \bm {H}_{-} + \bm {J}_i^T \bm {J}_i.
(11)
For exponential likelihoods, there is an asymptotic in-
verse relationship in the maximum likelihood estimator be-
tween the Hessian and covariance matrices [36]. Therefore,
our goal is to maximize the information I
 ·
obtained from
the i-th image, which can be found by minimizing the un-
certainty U
 ·
:
  \ tex
t
 
{
ar
g
} \ma x _
{
i
}
 \
m
a thc al 
{
I
}
 \b
e
g
in {pmatrix} \bm {H}_i \end {pmatrix} &= \text {arg}\min _{i} U \begin {pmatrix} \bm {\Sigma }_i \end {pmatrix} \nonumber \\ &= \text {arg}\min _{i} U \begin {pmatrix} \bm {H}_i^{-1} \end {pmatrix}.
(12)

<!-- page 5 -->
Table 1. Properties of P-Optimality for different values of p. Note: λk represent the eigenvalues of the covariance matrix Σi.
T-optimality
A-optimality
D-optimality
E-optimality
p
1
-1
0
∓∞
Equivalent
Formulae
1
l tr
 Σi

= 1
l
Pl
k λk
  1
l tr
 Hi
−1 =

1
l
Pl
k λ−1
k
−1
lp
|Σi| =
exp

1
l
Pl
k log λk

minλk
maxλk
Meaning
Average Variance
Harmonic Mean
Variance
Volume of covariance
hyper-ellipsoid
Single extreme
eigenvalue
In this work, we rely on the Theory of Optimal Experi-
mental Design [16, 30], which defines the P-Optimality un-
certainty metric as:
 U_p(\ b
m  
{ \Sigm
a }
_
i)  
=  \begin {pmatrix} \frac {1}{l} \text {trace} \begin {pmatrix} \bm {\Sigma }_i^p \end {pmatrix} \end {pmatrix}^{\frac {1}{p}},
(13)
where p is an integer. Depending on the chosen value for
p, the uncertainty function can have some special properties
[28], as detailed in Table 1.
3.3. Approximating the Covariance
In practice fitted 3D-GS models may contain millions of
parameters, which is intractable due to the cubic computa-
tional and quadratic memory complexity of computing the
eigenvalues. Therefore, we propose two approximations of
the Hessian matrix to save memory and computation.
Simple Diagonal: First, following the work of FisherRF
we propose a simple diagonal approximation of the covari-
ance matrix. FisherRF derives the approximation from a
Laplace approximation [6], where the covariance matrix is
approximated as the main diagonal plus a small regularizing
constant λθ:
  \ Sigma _\theta \approx \text {diag}(\Sigma _i) + \lambda _\theta , g
(14)
This formulation allows for an efficient and direct compari-
son of our method of computing information gain with Fish-
erRF, without consideration of correlation. Intuitively, the
constant λθ represents prior information for our method.
Block Diagonal: In order to capture some of the corre-
lation between 3D-GS parameters, we propose to approxi-
mate the Hessian matrix as a block diagonal matrix where
each block diagonal element contains the parameters of a
single ellipsoid. Please note that a block diagonal approxi-
mation has also been proposed within 3D-GS, however the
approximation was applied to the task of pruning 3D-GS
models to remove redundant ellipsoids [10]. Our insight is
that the parameters of the same ellipsoid are most likely to
be correlated, and a block diagonal matrix can be processed
in parallel on a GPU for efficient computation.
When constructing the block diagonal matrix, we note
that computing partial derivatives w.r.t. each pixel leads to
singularity issues since the partial derivative of each color
is the value α′
n for the ellipsoid. To avoid singularity issues,
we therefore compute the Hessian matrix by separately cal-
culating partial derivatives for each channel of every pixel,
instead of for every pixel for all channels. The block diago-
nal approximation is shown in Fig. 2.
3.4. Batch Selection
In practical applications it may be valuable to measure in-
formation gain over a set of candidate images, such as along
a trajectory or in keyframe selection. Following FisherRF,
we implement a simple approach which iteratively adds
or removes the most optimal candidate image, updates the
Hessian, and repeats the process without additional train-
ing. While simple, this batch information implementation
does capture the redundancy of views, as the change in the
Hessian is reflected when a candidate image is added.
4. Results
In this section, we study the effectiveness of our method on
quantifying information gained from obtaining new images.
Following the experimental setup of FisherRF, we quantita-
tively and qualitatively evaluate our method against base-
lines on the task of single view selection, where a 3D-GS
model is trained by iteratively selecting the most informa-
tive proposal view and fitting the model to a set of candidate
view. Next, we compare our method against the same base-
lines on batch view selection. Third, we compare the abil-
ity of each method to quantify uncertainty in novel views
by studying the correlation of information gain with recon-
struction metrics. Last, we perform ablation studies on the
parameters of the 3D-GS model to identify the most impor-
tant parameters for calculating information gain.
Baselines: We compare against the recently published
FisherRF [14], which calculates information gain as:
  \ma t hc
a
l {
I }(\bm 
{
H}_i) = \text {tr} \begin {pmatrix} (\bm {J}_i^T \bm {J}_i) \bm {H}^{-1}_{-} \end {pmatrix} 
(15)
with matrices modeled by the simple diagonal approxima-
tion. While FisherRF compared against a random and Ac-
tiveNeRF baseline [27], we omit these baselines as they
performed worse than FisherRF in their experiments, and
instead focus on comparing our results with FisherRF. We
also add a uniform sampling baseline, which is slightly dif-
ferent from the implementation of random in FisherRF that

<!-- page 6 -->
(a) Uniform Sampling
(b) FisherRF
(c) D-Opt. (Block)
(d) Ground Truth
Figure 4. Comparison of view selection methods on the Mip-Nerf360 dataset with 10 views. The columns are built by different methods
and in order are: uniform sampling, FisherRF, Block D-GS, and the ground truth image.
we found was incorrectly implemented. Instead, our uni-
form sampling baseline samples views uniformly from the
training set, which results in a high coverage of the views in
the test set, especially if the training and testing views are
similarly distributed. As one of the merits of our approach is
a more general solution with optimal experimental design,
we implement A, D, E, and T Optimality baselines for com-
parison. All baselines are compared over the peak signal-to-
noise ratio (PSNR), structural similarity index (SSIM) [37],
and LPIPS metrics [41].
Dataset: All methods are compared on two common
radiance field datasets. First is the Mip-NeRF360 dataset
[2], which is a real-world high-resolution dataset commonly
used in novel view synthesis literature as well as by Fish-
erRF. Mip-NeRF360 contains nine scenes, with five out-
door scenes and four indoor, which we average performance
over to obtain the final results. Following FisherRF and the
original 3D-GS paper, we train all models at resolutions of
1060×1600 pixels. Additionally, the prior information con-
stant is set to a value of λθ = 10−6 for all models. The
Mip-NeRF360 dataset contains complex scenes, where the
benefit of strong view selection models is clear. However,
the dataset also contains some noisy images which are dis-
tributed at random, and can impact the results.
Therefore, following FisherRF, we also evaluate all mod-
els on the Blender dataset [26], which contains eight high-
fidelity objects modeled synthetically. While the scenes in
this experiment are less complex, this dataset allows us to
study information gain quantification without the stochas-
ticity introduced by real-world noisy images.
Table 2. Results on Single View Selection with 10 Views on the
Mip-Nerf360 Dataset.
Method
PSNR (↑)
SSIM (↑)
LPIPS (↓)
Uniform Sampling
17.29
0.508
0.432
FisherRF
16.81
0.493
0.445
A-Opt. (Simple)
15.55
0.452
0.480
E-Opt. (Simple)
15.33
0.436
0.488
T-Opt. (Simple)
17.91
0.520
0.420
D-Opt. (Simple)
17.95
0.535
0.411
D-Opt. (Block)
18.15
0.548
0.401
4.1. Single View Selection
First, we compare all methods on single view selection,
where one candidate view is selected at a time. We follow
the experimental setup of FisherRF with minimal modifica-
tions, including evaluating models on the ability to select
both ten views and twenty views. Note that accurate infor-
mation quantification is more apparent with ten views due
to the limited amount of training information.
Ten Views: For the ten view setup, each method be-
gins with 2 training views, and is trained for 100v itera-
tions, where v is the number of training views. The method
then selects a single candidate view to add to the training
set, and repeats the process until v = 10, at which point
the model trains until a cumulative total of 10, 000 training
steps. Qualitative examples on the Mip-Nerf360 dataset are
shown in Fig. 4, and qualitative examples on the Blender
dataset are shown in Fig. 1.
First, we compare models on the Mip-NeRF360 dataset,
whose performance metrics can be found in Table 2. In this

<!-- page 7 -->
Table 3. Results on Single View Selection with 10 Views on the
Blender Dataset.
Method
PSNR (↑)
SSIM (↑)
LPIPS (↓)
Uniform Sampling
23.32
0.885
0.101
FisherRF
24.59
0.897
0.091
A-Opt. (Simple)
22.39
0.876
0.116
E-Opt. (Simple)
21.40
0.862
0.129
T-Opt. (Simple)
25.40
0.908
0.080
D-Opt. (Simple)
25.52
0.909
0.078
D-Opt. (Block)
25.41
0.908
0.078
experiment, FisherRF is slightly outperformed by the uni-
form sampling method. We would like to note that uniform
sampling is actually an effective method for this experimen-
tal setup, and will achieve a near optimal performance if the
test and train set are similarly distributed. By reframing
the approach to information gain within 3D-GS as optimal
experimental design, we find that both T-Optimality and D-
Optimality approaches improve significantly over the Fish-
erRF and uniform sampling baselines.
Additionally, the
block diagonal approximation significantly improves the
structural quality measured by SSIM and LPIPS metrics.
Next, we repeat the same set of experiments on the syn-
thetic dataset, shown in Table 3. Here, we find that Fish-
erRF performs significantly better than the uniform sam-
pling method, which may be due to the less complex syn-
thetic scenes, where FisherRF is able to more accurately
quantify information on a single object. Similar to before,
we find that both T-Optimality and D-Optimality methods
outperform the baselines by a wide margin. Across all three
metrics, simple and block D-Optimality perform similarly,
which we expect is due to the saturation of performance as
the approaches are near optimal.
Twenty Views: The twenty view experimental setup fol-
lows a very similar approach, of iteratively adding views
and training for 100v iterations before selecting the next
view. However, this setup begins with v = 4 training views,
adds images until v = 20, and trains until a cumulative total
of 21, 000 training steps. Experimental results on the MIP
dataset can be found in Table 4. Similar to the ten view
experiment, we find that uniform sampling slightly outper-
Table 4. Results on Single View Selection with 20 Views on the
Mip-Nerf360 Dataset.
Method
PSNR (↑)
SSIM (↑)
LPIPS (↓)
Uniform Sampling
20.86
0.616
0.408
FisherRF
20.89
0.608
0.416
A-Opt. (Simple)
18.62
0.558
0.452
E-Opt. (Simple)
19.57
0.580
0.433
T-Opt. (Simple)
21.07
0.615
0.409
D-Opt. (Simple)
21.09
0.624
0.406
D-Opt. (Block)
21.32
0.636
0.397
forms FisherRF while T and D Optimality achieve the high-
est performance with an improvement from the block diag-
onal approximation.
4.2. Batch View Selection
Next, we compare all methods on batch view selection,
where information gain is evaluated over several views si-
multaneously before training. This problem is more appli-
cable to real-world scenarios such as information gained
over a robot trajectory, or identification of the best set of
views of an object.
Iterative: In the first experiment on batch view selec-
tion we follow the experimental set-up of FisherRF. The
procedure is similar to single view selection, however mod-
els begin with 4 training views, are trained for 150v itera-
tions between view selections, and select 4 views at a time
until 20 views are obtained. All models are trained for a
cumulative total of 10, 000 training steps. Results on the
Mip-Nerf360 dataset are shown in Table 5, and similar to
previous experiments demonstrate superior performance of
D and T optimality. Additionally, the block diagonal ap-
proximation improves structural quality measured by SSIM
and LPIPS metrics. Note that due to the iterative view se-
lection and training, this experiment may not reward batch
view diversity, motivating our next experiment.
Keyframe Selection: To further study batch view se-
lection in a setting more similar to SLAM applications, we
compare each approach on keyframe selection. All methods
are provided the same pre-trained 3D-GS model and select
ten keyframes without replacement from a set of views. The
selected views are then used to re-train a new 3D-GS model,
which is evaluated and compared as a measure of batch in-
formation quantification. Results on the Blender dataset are
summarized in Table 6, demonstrating low performance of
FisherRF, A Optimality and E Optimality which select sim-
ilar views. Instead, T and D Optimality select informative
and different views resulting in a large performance gap.
4.3. Correlation with Render Quality
Intuitively, we expect information gain and rasterization
quality at view points to be inversely related. For instance,
if a 3D-GS model has only been trained on the front side of
Table 5. Results on Batch View Selection on Mip-Nerf360 dataset.
Method
PSNR (↑)
SSIM (↑)
LPIPS (↓)
Uniform Sampling
20.42
0.613
0.389
FisherRF
20.50
0.603
0.399
A-Opt. (Simple)
18.14
0.553
0.428
E-Opt. (Simple)
17.88
0.535
0.440
T-Opt. (Simple)
20.73
0.611
0.391
D-Opt. (Simple)
20.86
0.624
0.383
D-Opt. (Block)
20.79
0.631
0.378

<!-- page 8 -->
Table 6. Results on Keyframe Selection on Blender Dataset.
Method
PSNR (↑)
SSIM (↑)
LPIPS (↓)
Uniform Sampling
23.47
0.888
0.109
FisherRF
18.37
0.829
0.184
A-Opt. (Simple)
17.05
0.811
0.226
E-Opt. (Simple)
16.55
0.786
0.255
T-Opt. (Simple)
24.90
0.903
0.096
D-Opt. (Simple)
24.26
0.899
0.101
D-Opt. (Block)
24.53
0.902
0.099
a chair, view points from the back side would have poor ren-
derings while providing high information gain to the 3D-GS
model. Therefore, as another test of our proposed method
of quantifying information gain, we create a sparsification
plot [13] to study the inverse relationship between image
information gain and image render quality.
The sparsification plot in Fig. 5 is created by first train-
ing a 3D-GS model on ten randomly selected images for
2, 000 iterations. Each method, using the same random seed
and trained model, sorts candidate views by expected infor-
mation gain. At decile increments, the cumulative average
reconstruction quality of views is calculated and plotted for
each method. Reading from left to right, the plot indicates
the average reconstruction quality of the most informative
views. Due to the inverse nature between image uncertainty
and information gain, we would expect the most informative
candidate views (left) to have low reconstruction quality.
We can see from Fig. 5 that D-Opt. (Block) has a mono-
tomic relationship with the reconstruction quality, whereas
FisherRF has difficulty with some objects. To better un-
derstand the relative performance, we also include plots of
the Uniform Sampling and Oracle methods, where the Uni-
form Sampling method shows no correlation between the
selected images and the reconstruction quality as expected.
The Oracle baseline sorts the candidate views by the actual
PSNR value, representing a perfect baseline, with similar
performance to D-Opt. (Block).
4.4. Ablation Study
Last, we conclude with ablation studies on the most impor-
tant parameters for information gain quantification. While
(a) Ficus
(b) Drums
Figure 5. Correlation of expected information gain with PSNR of
candidate views on two objects in Blender dataset.
Table 7. Ablation study on D-Opt. (Block) parameters on Blender
Dataset with 10 images. The parameters are sh: spherical har-
monics, α: opacity, µ: location, R: rotation, and S: scale.
Parameters Removed
PSNR (↑)
SSIM (↑)
LPIPS (↓)
{sh }
25.52
0.9089
0.0784
{α }
25.56
0.9087
0.0778
{µ, R, S }
25.36
0.9074
0.0793
∅
25.41
0.9084
0.0776
we evaluated our methods with all parameters, reducing
the number of parameters can improve computational effi-
ciency. We compare the performance of D-Opt. (Block)
with different parameter combinations in Table 7 on the
task of single view selection with 10 views on the Blender
dataset. Peculiarly, we find that the geometric parameters
are important to quantifying information with few images
while the spherical harmonics do not result in a significant
difference, supporting the approach of PUP 3D-GS [10].
We expect that the opacity decreases performance since our
approach does not capture the cross-correlation of ellip-
soids. Additionally, we suspect that the spherical harmonics
may be more useful with a well fitted scene which already
has learned the geometric structure. Nonetheless, these re-
sults indicate that information may be evaluated with a min-
imal set of the geometric parameters, which can increase
inference speed. Concretely, on the ship scene from the
Blender dataset the memory and latency are as follows: 2.62
GB at 0.15 Hz for full block diagonal, 75.3 MB at 1.71 Hz
for block diagonal without spherical harmonics, and 44.4
MB at 12.16 Hz for the simple approximation. Note that
the cost of the simple approximation is the same as that of
FisherRF.
5. Conclusion
In this paper, we introduced a novel method for calculating
the information gain from images in 3D-GS which builds
on prior literature of P-Optimality. Information quantifica-
tion for 3D-GS is an important problem for evaluating un-
certainty in novel environments, selecting key frames for
SLAM algorithms, and next best view applications. Our
novel formulation leads to a general solution with a simple
and block diagonal information matrix approximation, with
computational and performance trade-offs. We demonstrate
quantitatively that formulating the information quantifica-
tion with T and D Optimality improves performance com-
pared to the state of the art, supporting results from prior
literature.
While our method achieves significant results
quantitatively and qualitatively, the simple and block diag-
onal approximations discard correlation between ellipsoids.
For future work we would like to investigate re-formulating
the problem to include inter-ellipsoid information, such as
from the structural similarity loss or the opacity parame-
ter.

<!-- page 9 -->
References
[1] Yaakov Bar-Shalom, X Rong Li, and Thiagalingam
Kirubarajan. Estimation with Applications to Track-
ing and Navigation: Theory, Algorithms and Software.
John Wiley & Sons, 2004. 4
[2] Jonathan T. Barron, Ben Mildenhall, Dor Verbin,
Pratul P. Srinivasan, and Peter Hedman.
Mip-
NeRF 360: Unbounded Anti-Aliased Neural Radiance
Fields.
In Proc. IEEE Conf. Comput. Vis. Pattern
Recog., pages 5460–5469, 2022. 6
[3] Henry Carrillo, Ian Reid, and Jos´e A. Castellanos.
On the comparison of uncertainty criteria for active
SLAM. In Proc. IEEE Int. Conf. Robot. and Automa-
tion, pages 2080–2087, 2012. 2, 3
[4] Yongbo Chen, Shoudong Huang, and Robert Fitch.
Active SLAM for mobile robots with area coverage
and obstacle avoidance. IEEE/ASME Trans. Mecha-
tronics, 25(3):1182–1192, 2020. 3
[5] Thomas M. Cover and Joy A. Thomas. Elements of
Information Theory, chapter 8, pages 243–259. John
Wiley & Sons, Ltd, 2005. 3
[6] Erik Daxberger, Agustinus Kristiadi, Alexander Im-
mer, Runa Eschenhagen, Matthias Bauer, and Philipp
Hennig.
Laplace redux – effortless Bayesian deep
learning. In Proc. Advances Neural Inform. Process.
Syst. Conf., 2021. 5
[7] Ben Fei, Jingyi Xu, Rui Zhang, Qingyuan Zhou, Wei-
dong Yang, and Ying He. 3D Gaussian Splatting as
New Era: A Survey. IEEE Trans. Graph., pages 1–20,
2024. 1, 2, 3
[8] Maani Ghaffari Jadidi, Jaime Valls Miro, and Gamini
Dissanayake. Gaussian processes autonomous map-
ping and exploration for range-sensing mobile robots.
Auton. Robot., 42(2):273–290, 2018. 2
[9] Lily Goli, Cody Reading, Silvia Sell´an, Alec Jacob-
son, and Andrea Tagliasacchi. Bayes’ Rays: Uncer-
tainty Quantification for Neural Radiance Fields. In
Proc. IEEE Conf. Comput. Vis. Pattern Recog., pages
20061–20070, 2024. 2
[10] Alex Hanson,
Allen Tu,
Vasu Singla,
Mayuka
Jayawardhana, Matthias Zwicker, and Tom Goldstein.
PUP 3D-GS: Principled Uncertainty Pruning for 3D
Gaussian Splatting. arXiv, abs/2406.10219, 2024. 2,
5, 8
[11] Xun Huan and Youssef M. Marzouk.
Simulation-
based optimal Bayesian experimental design for non-
linear systems. J. of Comput. Phys., 232(1):288–317,
2013. 2
[12] Xun Huan and Youssef M. Marzouk. Gradient-based
stochastic optimization methods in Bayesian experi-
mental design. Int. J. for Uncertainty Quant., 4(6):
479–510, 2014. 2
[13] Eddy Ilg, ¨Ozg¨un C¸ ic¸ek, Silvio Galesso, Aaron Klein,
Osama Makansi, Frank Hutter, and Thomas Brox. Un-
certainty Estimates and Multi-hypotheses Networks
for Optical Flow. In Proc. European Conf. Comput.
Vis., pages 677–693, 2018. 8
[14] Wen Jiang, Boshu Lei, and Kostas Daniilidis. Fish-
erRF: Active View Selection and Uncertainty Quan-
tification for Radiance Fields using Fisher Informa-
tion.
In Proc. European Conf. Comput. Vis., pages
422–440, 2024. 2, 5
[15] Bernhard
Kerbl,
Georgios
Kopanas,
Thomas
Leimk¨uhler, and George Drettakis.
3D Gaussian
Splatting for Real-Time Radiance Field Rendering.
IEEE Trans. Graph., 42(4), 2023. 1, 2, 3
[16] Jack Kiefer. General equivalence theory for optimum
designs (approximate theory). The annals of Statistics,
pages 849–879, 1974. 2, 3, 5
[17] Andreas Kirsch and Yarin Gal. Unifying Approaches
in Active Learning and Active Sampling via Fisher
Information and Information-Theoretic Quantities. J.
Mach. Learning Res., 2022. 2
[18] S. Kullback and R. A. Leibler. On Information and
Sufficiency. The Ann. Math. Stat., 22(1):79 – 86, 1951.
2
[19] Ruiqi Li and Yiu ming Cheung.
Variational Multi-
scale Representation for Estimating Uncertainty in 3D
Gaussian Splatting. In Proc. Advances Neural Inform.
Process. Syst. Conf., 2024. 2
[20] D. V. Lindley. On a Measure of the Information Pro-
vided by an Experiment. The Ann. Math. Stat., 27(4):
986 – 1005, 1956. 2
[21] D. V. Lindley. On a Measure of the Information Pro-
vided by an Experiment. The Ann. Math. Stat., 27(4):
986–1005, 1956. 2
[22] Guangyi Liu, Wen Jiang, Boshu Lei, Vivek Pandey,
Kostas Daniilidis, and Nader Motee. Beyond Uncer-
tainty: Risk-Aware Active View Acquisition for Safe
Robot Navigation and 3D Scene Understanding with
FisherRF. arXiv, abs/2403.11396, 2024. 2
[23] Ruben Martinez-Cantin,
Nando de Freitas,
Eric
Brochu, Jos´e A. Castellanos, and A. Doucet.
A
bayesian exploration-exploitation approach for op-
timal online sensing and planning with a visually
guided mobile robot. Auton. Robot., 27:93–103, 2009.
2
[24] Hidenobu Matsuki, Riku Murai, Paul H.J. Kelly, and
Andrew J. Davison. Gaussian Splatting SLAM. In
Proc. IEEE Conf. Comput. Vis. Pattern Recog., pages
18039–18048, 2024. 1, 2
[25] Lyudmila Mihaylova, Tine Lefebvre, Herman Bruyn-
inckx, Klaas Gadeyne, and Joris De Schutter. A com-
parison of decision making criteria and optimization
methods for active robotic sensing.
In Numerical

<!-- page 10 -->
Methods and Applications, pages 316–324. Springer,
2003. 3
[26] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tan-
cik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren
Ng. NeRF: Representing Scenes as Neural Radiance
Fields for View Synthesis. In Proc. European Conf.
Comput. Vis., pages 405–421, 2020. 2, 6
[27] Xuran Pan, Zihang Lai, Shiji Song, and Gao Huang.
ActiveNeRF: Learning Where to See with Uncertainty
Estimation.
In Proc. European Conf. Comput. Vis.,
pages 230–246, 2022. 2, 5
[28] Julio A Placed and Jos´e A Castellanos. A General Re-
lationship between Optimality Criteria and Connectiv-
ity Indices for Active Graph-SLAM. IEEE Robot. Au-
tom. Letter., 8(2):816–823, 2022. 3, 5
[29] Julio A Placed, Juan J G´omez Rodr´ıguez, Juan D
Tard´os, and Jos´e A Castellanos. Explorb-slam: Ac-
tive visual slam exploiting the pose-graph topol-
ogy. In Iberian Robotics conference, pages 199–210.
Springer, 2022. 2, 3
[30] Julio A. Placed, Jared Strader, Henry Carrillo, Nikolay
Atanasov, Vadim Indelman, Luca Carlone, and Jos´e A.
Castellanos. A Survey on Active Simultaneous Local-
ization and Mapping: State of the Art and New Fron-
tiers. IEEE Trans. Robot., 39:1686–1705, 2022. 2, 3,
5
[31] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian
Wang, and Hanspeter Pfister.
LangSplat: 3D Lan-
guage Gaussian Splatting. In Proc. IEEE Conf. Com-
put. Vis. Pattern Recog., pages 20051–20060, 2024. 1,
2
[32] Mar´ıa L. Rodr´ıguez-Ar´evalo, Jos´e Neira, and Jos´e A.
Castellanos. On the Importance of Uncertainty Rep-
resentation in Active SLAM. IEEE Trans. Robot., 34
(3):829–834, 2018. 3
[33] Robert Sim and Nicholas Roy. Global a-optimal robot
exploration in slam. In Proc. IEEE Int. Conf. Robot.
and Automation, pages 661–666. IEEE, 2005. 2, 3
[34] Matthew Strong, Boshu Lei, Aiden Swann, Wen Jiang,
Kostas Daniilidis, and Monroe Kennedy III au2. Next
Best Sense: Guiding Vision and Touch with FisherRF
for 3D Gaussian Splatting.
arXiv, abs/2410.04680,
2024. 2
[35] Matthew Tancik,
Vincent Casser,
Xinchen Yan,
Sabeek Pradhan, Ben Mildenhall, Pratul P. Srinivasan,
Jonathan T. Barron, and Henrik Kretzschmar. Block-
NeRF: Scalable Large Scene Neural View Synthe-
sis. In Proc. IEEE Conf. Comput. Vis. Pattern Recog.,
pages 8248–8258, 2022. 2
[36] A. W. van der Vaart. Asymptotic Statistics, chapter 4,
page 35–40. Cambridge University Press, 1998. 4
[37] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Si-
moncelli. Image quality assessment: from error visi-
bility to structural similarity. IEEE Trans. Image Pro-
cess., 13(4):600–612, 2004. 3, 6
[38] Joey Wilson, Marcelino Almeida, Min Sun, Sachit
Mahajan, Maani Ghaffari, Parker Ewen, Omid Ghase-
malizadeh, Cheng-Hao Kuo, and Arnie Sen. Modeling
Uncertainty in 3D Gaussian Splatting through Con-
tinuous Semantic Splatting. ArXiv, abs/2411.02547,
2024. 1
[39] Joey Wilson, Yuewei Fu, Joshua Friesen, Parker
Ewen, Andrew Capodieci, Paramsothy Jayakumar,
Kira Barton, and Maani Ghaffari.
ConvBKI: Real-
Time Probabilistic Semantic Mapping Network With
Quantifiable Uncertainty.
IEEE Trans. Robot., 40:
4648–4667, 2024. 1
[40] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang
Wang, Dong Wang, and Xuelong Li.
GS-SLAM:
Dense Visual SLAM with 3D Gaussian Splatting. In
Proc. IEEE Conf. Comput. Vis. Pattern Recog., pages
19595–19604, 2024. 2
[41] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli
Shechtman, and Oliver Wang. The Unreasonable Ef-
fectiveness of Deep Features as a Perceptual Metric. In
Proc. IEEE Conf. Comput. Vis. Pattern Recog., pages
586–595, 2018. 6
[42] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen
Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya
You, Zhangyang Wang, and Achuta Kadambi. Feature
3dgs: Supercharging 3d gaussian splatting to enable
distilled feature fields. In Proc. IEEE Conf. Comput.
Vis. Pattern Recog., pages 21676–21685, 2024. 1, 2
[43] Liyuan Zhu, Yue Li, Erik Sandstr¨om, Shengyu Huang,
Konrad Schindler, and Iro Armeni. LoopSplat: Loop
Closure by Registering 3D Gaussian Splats.
arXiv,
abs/2408.10154, 2024. 1, 2

<!-- page 11 -->
POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality
Supplementary Material
Table 8. Results on Single View Selection with 20 Views on the
Blender Dataset.
Method
PSNR (↑)
SSIM (↑)
LPIPS (↓)
Uniform Sampling
26.15
0.918
0.084
FisherRF
27.12
0.925
0.079
A-Opt. (Simple)
24.88
0.908
0.094
E-Opt. (Simple)
24.87
0.901
0.097
T-Opt. (Simple)
27.29
0.929
0.076
D-Opt. (Simple)
27.25
0.930
0.075
D-Opt. (Block)
27.28
0.930
0.075
Table 9. Results on Batch View Selection on Blender dataset.
Method
PSNR (↑)
SSIM (↑)
LPIPS (↓)
Uniform Sampling
26.64
0.925
0.074
FisherRF
27.64
0.932
0.069
A-Opt. (Simple)
25.61
0.916
0.082
E-Opt. (Simple)
24.99
0.908
0.087
T-Opt. (Simple)
27.89
0.936
0.065
D-Opt. (Simple)
27.87
0.937
0.064
D-Opt. (Block)
27.80
0.935
0.065
6. Twenty View Blender Results
Due to page constraints, we were unable to report the results
of each method on twenty view selection of the Blender
dataset with the single or batch view schemes. In the main
paper, we focused on ten views with the Blender dataset, as
we find that the results saturate when more views are added.
Single view selection results are shown in Table 8, where
FisherRF, T and D optimality achieve significantly higher
performance than uniform sampling. T and D optimality
outperform FisherRF, and have similar saturated results to
each other.
Results on the batch view selection are demonstrated in
Table 9, where we find similar results as the twenty view
experiment. FisherRF, T and D optimality outperform the
other baselines with T and D optimality achieving the high-
est results. Due to the large number of views and simple
scenes, results are saturated between T and D optimality.
7. Keyframe Selection
Next we present results from the keyframe selection ex-
periment on the Mip-Nerf360 dataset. In this experiment,
we find a large performance gap between FisherRF, A and
E Optimality and the other methods similar to with the
Table 10. Results on Keyframe Selection on Mip-Nerf360 Dataset.
Method
PSNR (↑)
SSIM (↑)
LPIPS (↓)
Uniform Sampling
18.30
0.560
0.435
FisherRF
15.66
0.471
0.515
A-Opt. (Simple)
15.67
0.479
0.519
E-Opt. (Simple)
15.96
0.475
0.524
T-Opt. (Simple)
18.66
0.560
0.425
D-Opt. (Simple)
18.57
0.559
0.426
D-Opt. (Block)
18.73
0.571
0.417
Blender dataset. T and D optimality outperform the uni-
form sampling baseline, however results are saturated with
ten well-chosen views. Despite the performance saturation,
the block diagonal approximation leads to a noticeable im-
provement in SSIM and LPIPS metrics.
8. Render Quality Correlation on All Scenes
In this section, we first provide a more detailed explanation
of the experimental setup for our study of the correlation
between information gain and render quality. Next, we pro-
vide the sparsification plots for the remaining objects in Fig.
6.
In addition to identifying the most important training
images, another key problem for applying 3D-GS to real-
world applications is uncertainty quantification. Since in-
formation gain is dependent on the amount of information
already present in a trained 3D-GS model at a candidate
view, we would expect an inverse relationship between in-
formation gain and render quality.
Therefore, we lever-
age sparsification plots to study the correlation between
information gain and render quality. Intuitively, if a 3D-
GS model has already observed data similar to a view, the
method should quantify small information gain and the ren-
der should have a high reconstruction quality. Similarly,
a candidate view with high information gain implies the
model has limited information on the viewpoint, and the
render would likely have poor reconstruction quality. This
also follows from the inverse relationship between uncer-
tainty and information stated in Section 3.2.1.
In order to study this relationship, we leverage sparsifica-
tion plots. The primary idea behind sparsification plots is to
sort candidate views by information gain, and observe the
relationship between information gain at candidate views
and render quality at the candidate views. In this experi-
ment, we train a single 3D-GS model on ten randomly cho-
sen views so that there is new information at the remaining

<!-- page 12 -->
(a) Chair
(b) Hotdog
(c) Lego
(d) Materials
(e) Mic
(f) Ship
Figure 6. Uncertainty correlation plots on all remaining scenes of the Blender dataset. The oracle represents a perfect sorting of the views
by PSNR. If the information gained by candidate views is well calibrated, the ordering should be similar to that of the oracle, resulting in
a low value at the left of the plot which contains the average reconstruction quality of the most informative views.
views in the dataset. Next, each method sorts the remaining
views by estimated information gain from most information
to least information. The purpose of the sparsification plot
is then to study how the views are sorted by estimated in-
formation gain.
To create the sparsification plot, the sorted views are or-
ganized into groups based on decile. For example, for one
hundred candidate views the first group would contain the
ten most informative views and the final group would con-
tain the ten least informative views. Next, the groups are
combined iteratively beginning from the most informative
views and the average PSNR is calculated for the combined
groups. Therefore, the plot represents the average PSNR of
the x% most informative views. As a result, we would ex-
pect to see a low PSNR for the most informative views at the
left of the plot, and all methods converge at the right of the
plot when calculating the average PSNR over all images.
For baselines we use the uniform sampling method,
which should demonstrate no correlation between expected
information gain and average PSNR. We also introduce an
oracle baseline which directly observes the PSNR of each
render and represents an ideal ordering. We compare Fish-
erRF and D-Opt. (Block) with the baselines, where the best
performing method is the method most similar to the oracle
demonstrating a relationship between uncertainty and ren-
der quality.
Fig. 6 details the remaining plots for all objects in the
dataset.
In the main text, we chose figures which high-
lighted the performance of FisherRF as well as D-Opt.
(Block). D-Opt. (Block) generally has a monotomic behav-
ior and performance near the oracle. However, FisherRF
sometimes does not exhibit the same behavior depending
on the object. We note that this plot is not the intended goal
of FisherRF, however we would expect a strong correlation
between information gain and reconstruction error as stated
previously.
