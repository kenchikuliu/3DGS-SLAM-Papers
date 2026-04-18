# 3D Gaussian Point Encoders

Jim James Georgia Tech jimjames@gatech.edu

Ben Wilson Georgia Tech

Simon Lucey University of Adelaide

James Hays Georgia Tech

## Abstract

In this work, we introduce the 3D Gaussian Point Encoder, an explicit per-point embedding built on mixtures of learned 3D Gaussians. This explicit geometric representation for 3D recognition tasks is a departure from widely used implicit representations such as PointNet. However, it is difficult to learn 3D Gaussian encoders in end-toend fashion with standard optimizers. We develop optimization techniques based on natural gradients and distillation from PointNets to find a Gaussian Basis that can reconstruct PointNet activations. The resulting 3D Gaussian Point Encoders are faster and more parameter efficient than traditional PointNets. As in the 3D reconstruction literature where there has been considerable interest in the move from implicit (e.g., NeRF) to explicit (e.g., Gaussian Splatting) representations, we can take advantage of computational geometry heuristics to accelerate 3D Gaussian Point Encoders further. We extend filtering techniques from 3D Gaussian Splatting to construct encoders that run 2.7Ã faster as a comparable accuracy PointNet while using 46% less memory and 88% fewer FLOPs. Furthermore, we demonstrate the effectiveness of 3D Gaussian Point Encoders as a component in Mamba3D, running 1.27Ã faster and achieving a reduction in memory and FLOPs by 42% and 54% respectively. 3D Gaussian Point Encoders are lightweight enough to achieve high framerates on CPU-only devices. Code is available at https://github.com/ jimtjames/3dGaussianPointEncoders

## 1. Introduction

Point cloud processing plays a crucial role in robotics and autonomous vehicles where LiDAR and related sensors capture three-dimensional spatial data. Since point clouds are unordered sets of points, deep networks designed for point cloud analysis must be permutation invariant to ensure consistent representations regardless of input ordering. Methods such as PointNet achieve this by employing symmetric aggregation functions, preserving the inherent structure of the data while enabling effective learning.

This key feature has made PointNet ubiquitous in a variety of 3D tasks, including: classification [21, 26], detection [19, 38, 45], and segmentation.

In PointNet, the majority of computational cost arises from per-point embedding, as it requires computing multiple large MLPs across a high number of points in each input point cloud. In contrast, the classifier stage applies MLPs only to a single global feature, making it relatively lightweight. To address this inefficiency, prior works have explored alternative approaches [43], such as LUTI-MLP [28], which replaces computationally expensive ReLU-MLP operations with lookup tables, and GPointNet [29], which employs single Gaussians. Although these methods greatly reduce FLOPs per sample compared to PointNet, their throughput on low-power platforms, such as CPU inference, remains limited. LUTI-MLP suffers from complex memory access patterns, while GPointNet requires evaluating a large number of Gaussian kernels, both of which hinder performance gains in resource-constrained environments.

Recently, explicit models based on mixtures of 3D Gaussians have gained traction in the view-synthesis literature due to their ability to efficiently represent volumetric data [5, 8, 14, 17]. Several studies have leveraged the explicit nature of 3D Gaussians to reduce computational costs, employing techniques such as Gaussian pruning [8, 15, 22] and heuristic-based filtering of low-value Gaussian-point pairs [14, 39]. These optimizations significantly accelerate inference compared to per-point coordinate networks.

In this work, we propose a novel 3D Gaussian Point Encoder, a per-point embedding that integrates PointNetâs max-pooling aggregation with performance optimizations from view synthesis using mixtures of 3D Gaussians. By interpreting each dimension of PointNetâs embedding function as a volumetric representation, we leverage the capacity of 3D Gaussian mixtures to model volumes, enabling a lightweight approximation of a pre-trained PointNet. Moreover, we demonstrate it is possible to train this encoder endto-end through Gaussian-specific natural gradient methods. Additionally, we exploit the explicit structure of Gaussians to enhance computational efficiency through filtering Gaussian-Point pairs. To assess the effectiveness of our approach, we conduct shape classification experiments on ModelNet40 [36] and ScanObjectNN [31] while equipping our encoder with classical and modern classifiers from PointNet and Mamba3D. In summary, our primary contributions are:

(i) We present a novel explicit 3D representation as a drop in replacement for the implicit PointNet representations which are ubiquitous in 3D scene understanding

(ii) We discover that 3D Gaussian representations present significant optimization challenges when using offthe-shelf optimizers. We find two paths to overcome this roadblock â distillation from PointNet teachers and direct optimization with Natural Gradients

(iii) We show that explicit representations can benefit from geometric acceleration techniques, such as pairwise Gaussian-point filtering, inspired by the 3DGS literature

(iv) We demonstrate that 3D Gaussian representations can achieve similar levels of accuracy to PointNet per-point embeddings, while achieving 2.7Ã higher throughput and 46% less memory. When integrated into Mamba3D, we achieve 1.27Ã the throughput and 42% less memory.

## 2. Related Work

Point Embeddings. PointNet [25] is one of the first models to directly process point clouds, utilizing a per-point MLP with ReLU activations followed by a max-pooling operator. The output of this MLP serves as a spatial encoding for each point, while max-pooling aggregates these per-point embeddings into a single global feature representing the entire point cloud. Several works have extended PointNet to support hierarchical feature learning, including PointNet++ [25] and its modern variants [21, 27].

Several approaches rely on Transformers [32] as a component of their backbone, such as Point Cloud Transformer [11] and Point Transformer [34, 35, 42]. Transformer methods have the advantage of being possible to train from the vast quantity of unlabeled data via self-supervised learning, as done in Point-MAE [23] and Point-BERT [40]. However, Transformers suffer from quadratic time complexity in sequence length, potentially resulting in inefficiency when processing large point sets. To resolve this issue, recent approaches instead utilize Mamba [4, 10], a structured state space model alternative to the Transformer with linear time complexity. PointMamba [20] and PCM [41] aim to produce a vanilla Mamba-based model without a hierarchical encoder architecture. Most recently, Mamba3D [13] achieves near state-of-the-art performance on point classification, expanding upon PointMamba through the use of a bidirectional Mamba variant and local feature aggregation.

Efficient PointNet Variants. A variety of prior works have explored efficient point encoders based on PointNetâs per-point embedding with max-pooling framework. LUTI-MLP [28] utilizes a lookup table per dimension of Point-Netâs embeddings formed followed by trilinear interpolation to form point embeddings. The lookup table is optimized during training time by discretizing and interpolating a pre-trained PointNet MLP, which is then voxelized at test time. This results in faster calculation of point embeddings compared to PointNetâs MLP for 3D point clouds. However, as the input dimension increases, the runtime and memory cost grows exponentially due to the increased lookup table size and number of neighbors to interpolate. GPointNet [29] instead represents each dimension of a point embedding via the likelihood of a single anisotropic Gaussian, resulting in an encoder requiring significantly fewer FLOPs per sample compared to PointNet.

Preconditioning and Natural Gradients. Preconditioning is a technique in which an optimization problem is transformed to make it more amenable to numerical solvers. Several optimizers internally apply preconditioning to their gradients to stabilize training. These include the diagonal preconditioners in AdaGrad [6] and Adam [18], as well as the block diagonal preconditioners in modern optimizers such as Shampoo [12] and SOAP [33]. One explicit form of preconditioning is Natural Gradients [1], a generalization of steepest descent for arbitrary metric spaces. This in contrast to standard gradient descent, where steps are considered with fixed Euclidean distance. Amari [1] demonstrated that given a metric, Natural Gradient descent can be viewed as preconditioning the gradients by the inverse of the metric spaceâs Riemannian metric tensor.

Mixtures of Gaussians as Approximators. Methods utilizing mixtures of Gaussians, or more generally, radial basis functions [24], have been widely studied. Classical works have used isotropic Gaussians to approximate volumes [44]. Most recently, a variety of works involving Gaussians have been applied to novel view synthesis. 3D Gaussian Splatting (3DGS) [17] represents volumes via a mixture of anistropic Gaussians, and is able to render novel views signficantly faster than coordinate networks. The use of explicit Gaussians allows the method to exploit sparsity in real-world scenes. However, optimizing the set of Gaussians requires additional techniques compared to coordinate neural network-based approaches. Niemeyer et al. [22] notes that utilizing guidance from a pre-trained coordinate network can help train a more robust Gaussian representation to work around this issue.

Several works [7, 8, 15, 22] have reduced the computational costs of 3DGS via pruning and filtering. Ye et al. [39] improve runtime by learning a truncation threshold on the Mahalanobis distance for each Gaussian, while Hanson et al. [14] instead propose filtering before the Mahalanobis distance calculation by bounding each Gaussian with a rectangle or bounding via tiles, and then only computing points that fall within each bounding box or tile respectively.

## 3. Method

We introduce the 3D Gaussian Point Encoder (3DGPE), which replaces PointNet representations by simple, explicit 3D Gaussian functions for effective 3D shape classification. Surprisingly, we find that distilling point cloud features into a Gaussian-based network yields superior performance compared to directly optimizing Gaussian parameters. Additionally, our 3D Gaussian representation significantly reduces computational overhead by efficiently removing Gaussians and Gaussian-point pairs that do not meaningfully contribute to the final feature representation.

Our 3D Gaussian Point Encoder comprises two key components: the Gaussian Basis Encoder and the Gaussian Basis Mixer. The Gaussian Basis Encoder encodes a pointâs representation by computing its proximity to a set of 3D Gaussians, effectively capturing local geometric features. The Gaussian Basis Mixer then integrates these Gaussian-based features, transforming them into a richer and more expressive feature representation. This structured approach enables efficient and flexible encoding of spatial information for downstream tasks. In the following section, we outline their construction.

## 3.1. Gaussian Basis Encoder

The Gaussian Basis Encoder is a parametric function that maps input points from a 3D point cloud into a structured feature space using a set of learnable Gaussian functions. Given a point cloud $\mathcal { X } = \{ x _ { i } \} _ { i = 1 } ^ { N }$ , where each point $x _ { i } \in \mathbb { R } ^ { 3 }$ , the encoder represents the input as a mixture of spatial Gaussians. Each Gaussian component g is defined by a mean $\mu _ { g } \in \mathbb { R } ^ { 3 }$ , which represents the center of the Gaussian in 3D space; a precision matrix (inverse of covariance matrix) $\pmb { \Sigma } _ { g } ^ { - 1 } \in \mathbb { R } ^ { 3 \times 3 }$ , modeling spatial extent; and a set of mixture coefficients $\{ \alpha _ { g , k } \} _ { k = 1 } ^ { K }$ , where K denotes the number of activation volumes.

Covariance Parameterization. To ensure that $\Sigma _ { g } ^ { - 1 }$ remains positive semi-definite, we parameterize it using the Cholesky decomposition [5]:

$$
\begin{array} { r } { \pmb { \Sigma } _ { g } ^ { - 1 } = \mathbf { L } _ { g } \mathbf { L } _ { g } ^ { \top } , } \end{array}\tag{1}
$$

where $\mathbf { L } _ { g }$ is a lower triangular matrix. This factorization guarantees valid covariance matrices while enabling efficient optimization. We parameterize the inverse directly to reduce the risk of numerical instability during training.

Feature Encoding. For each input point x, we compute its unweighted Gaussian likelihood under each Gaussian g as follows:

$$
\phi _ { g } ( x ) = \exp \left( - \frac { 1 } { 2 } ( x - \pmb { \mu } _ { g } ) ^ { \top } \pmb { \Sigma } _ { g } ^ { - 1 } ( x - \pmb { \mu } _ { g } ) \right) .\tag{2}
$$

This function measures the proximity of x to the Gaussian distribution centered at $\mu _ { g }$ , with spatial spread determined by $\Sigma _ { g }$

## 3.2. Gaussian Basis Mixer

Following the Gaussian Basis Encoder, we introduce the Gaussian Basis Mixer, a critical component of our architecture that distinguishes it from prior methods such as GPoint-Net [29]. Unlike previous approaches, the Gaussian Basis Mixer employs shared Gaussians across multiple activation volumes, effectively utilizing these Gaussians as basis functions. This design exploits redundancy, enhancing efficiency and enabling the network to represent complex activation volumes beyond simple ellipsoids.

Mathematically, the Gaussian Basis Mixer applies a linear layer using mixture coefficients to combine Gaussians and form activation volumes:

$$
l _ { k } ( x ) = \sum _ { g = 1 } ^ { N _ { G } } \alpha _ { g , k } \phi _ { g } ( x ) + b _ { k } ,\tag{3}
$$

where $b _ { k }$ is a bias term unique to each activation volume. Following this, we maxpool across points to produce a permutation-invariant global feature.

Gaussian sharing significantly reduces latency and memory overhead as the input dimension increases, addressing a key computational bottleneck. The complexity of computing Gaussian likelihoods grows quadratically with input dimension due to the Mahalanobis distance computation. In contrast, the additional cost of uniquely recombining Gaussians for each activation volume scales only linearly with both the total number of Gaussians $( N _ { G } )$ and the number of activation volumes (K). This trade-off enables our architecture to efficiently handle high-dimensional inputs while maintaining expressiveness.

## 3.3. Shape Classification Architecture

We primarily experiment with utilizing our encoder with two classification architectures: PointNet [25] and Mamba3D [13].

3D Gaussian Point Encoder with PointNet. The 3D Gaussian Point Encoder serves as the per-point embedding network; however, we add a few critical components to mimic a PointNet. The T-Net used in PointNet predicts a rotation matrix to achieve invariance to geometric transformations such as translation, rotation, and scaling. Since itâs constructed from a PointNet, we are able to replace it with a 3DGPE network. We add the 3D Gaussian T-Net prior to passing the points through the backbone network. After then generating the global feature from the 3DGPE network, we compute our class logits by passing the global feature through a simple MLP classifier. This is equivalent in architecture to PointNetâs classifier.

<!-- image-->  
Figure 1. Base architecture of 3DGPE. An input point cloud is first pre-processed, such as by a T-Net or through Farthest Point Sampling and KNN. Afterwards, each input point is processed independently through the Gaussian Basis Encoder by first computing a set of Gaussian likelihoods, followed by the Gaussian Basis Mixer, mixing the likelihoods to form a set of embeddings for each activation volume. We max-pool across points to derive a global feature which is then passed to a downstream classifier, such as an MLP.

<!-- image-->  
Figure 2. Implicit to Explicit 3D Knowledge Distillation. Points are sampled and pre-processed (T-Net or FPS + KNN) before being passed through each encoder. We then measure $L _ { 1 }$ loss between the 3D Gaussian Point Encoder and PointNet per-point embeddings. Maroon outlines indicate trainable components, while blue indicates frozen components.

3D Gaussian Point Encoder with Mamba3D. Here, the 3D Gaussian Point Encoder serves as the patch encoder, generating feature embeddings for point sets formed through farthest point sampling and KNN-based grouping. After the point patches have been passed through the 3D Gaussian Point Encoder, we pass these embeddings through Mamba3Dâs middle encoder blocks while applying positional encodings. These blocks consist of a per-group normalization and feature aggregation operation, followed by a bi-directional state space model to capture global information about the point patch embeddings. We then compute class logits by applying an MLP classifier to the aggregated point embeddings. See Han et al. [13] for more details.

## 3.4. Optimization of 3D Gaussian Point Encoder

We observe that end-to-end training of the 3D Gaussian Point Encoder with standard optimizers yields significantly lower performance compared to baseline models in Point-Net and Mamba3D, as shown in Tab. 2. We uncover two strategies to bypass this roadblock. The first is preconditioning the gradient via natural gradients, and the second is to distill the implicit geometry of PointNet features to the explicit geometry of 3DGPE.

## 3.4.1. Natural Gradients for 3D Gaussians

Standard gradient descent minimizes loss by stepping in the direction of steepest decrease in the loss, assuming a fixed step size in Euclidean distance. Natural gradients [1] generalize this by considering a different metric for step size, often resulting in faster convergence. Amari [1] notes that natural gradient descent can be performed via SGD while preconditioning the gradients by the inverse of the Riemannian metric tensor associated with a given parameter space, like so:

$$
\mathbf { x } ^ { t + 1 } = \mathbf { x } ^ { t } - \gamma \mathbf { G } ^ { - 1 } \nabla \mathcal { L } ( \mathbf { x } ^ { t } ) ,\tag{4}
$$

where $x ^ { t }$ is a parameter at iteration $t , \gamma$ is the learning rate, G is the Riemannian metric tensor, and L is the loss function. Gaussians as primitives admit two natural metrics.

Mahalanobis Distance. Gaussian likelihoods are a function of the Mahalanobis distance of a query point to the mean with respect to the precision matrix. Accordingly, we can consider each Gaussianâs mean an element in the metric space equipped with the Mahalanobis distance given its precision matrix. In this case, the Riemannian metric tensor is the precision matrix itself [16]. Thus, the natural gradient update for the g-th Gaussianâs mean becomes:

$$
\pmb { \mu } _ { g } ^ { t + 1 } = \pmb { \mu } _ { g } ^ { t } - \gamma \pmb { \Sigma } _ { g } ^ { t } \nabla \mathcal { L } \left( \pmb { \mu } _ { g } ^ { t } \right) .\tag{5}
$$

See Sec. 1 of the supplemental material for an example.

Fisher Information Metric. If we view each Gaussian primitive as a probability distribution, we can treat its mean and Cholesky decomposition parameters combined as an element in a parameter space defining probability distributions. One commonly used divergence for comparing such distributions is the KL divergence. The KL divergence can be approximated via a second order Taylor expansion to Fisher Information [30]. In this case, the Riemannian metric tensor is the Fisher Information Matrix $F _ { g } ,$ whose inverse can be easily computed in closed form and includes the same mean update as the Mahalanobis case [30]. Thus, the natural gradient update for the g-th Gaussianâs parameters becomes:

$$
\left( \pmb { \mu } _ { g } ^ { t + 1 } , \pmb { L } _ { g } ^ { t + 1 } \right) = \left( \pmb { \mu } _ { g } ^ { t } , \pmb { L } _ { g } ^ { t } \right) - \gamma \pmb { F } _ { g } ^ { - 1 } \nabla \mathcal { L } \left( \pmb { \mu } _ { g } ^ { t } , \pmb { L } _ { g } ^ { t } \right) .\tag{6}
$$

In comparison to preconditioners applied by AdaGradinspired optimizers [6, 18], neither of these preconditioning matrices are constrained to be diagonal, allowing them to capture more of the local geometry of each Gaussian. Furthermore, while optimizers like Shampoo [12] and SOAP [33] instead utilize block diagonal preconditioning matrices, they recalculate the preconditioning matrices only for a subset of gradient updates for efficiency.

## 3.4.2. Implicit to Explicit 3D Knowledge Distillation

Our second approach is to first directly supervise our 3D Gaussian Point Encoder via a pre-trained PointNet-style per-point embedding. For our PointNet classification experiments, we perform this in three stages. Initially, we optimize the first Gaussian Basis Encoder, which serves as a T-Net, by sampling random points from the minimum bounding rectangular prism of the training set (e.g., the unit cube). We then minimize the $L _ { 1 }$ loss between the per-point embeddings of PointNetâs T-Net and those generated by the Gaussian Basis Encoder, prior to maxpooling.

Next, we enhance the 3D Gaussian Basis T-Net by incorporating a copy of the transform regressor MLP from the pre-trained T-Net. Once the per-point encodings for the T-Net are aligned, we proceed to optimize the main Gaussian Basis Encoder, which replaces the PointNet encoder. This optimization follows a similar process: sampling points from the bounding rectangular prism, computing the transformed points via each encoderâs T-Net, and minimizing the $L _ { 1 }$ loss between the per-point embeddings produced by each encoder.

After this distillation phase, the model is trained end-toend on the training set, utilizing a copy of the parent modelâs classifier. To preserve the learned per-point embeddings, we apply a reduced learning rate to both the T-Net encoder and the main encoder parameters, preventing significant deviations in their representations.

For our Mamba3D classification experiments, we only need two stages, as there is only one per-point embedding to distill. We optimize our 3DGPE network, which serves as a patch encoder, by instead sampling point clouds from the training dataset, and applying farthest point sampling and KNN-based grouping to generate in-distribution point patches. Similar to the PointNet case, we aim to minimize $L _ { 1 }$ loss of the encodersâ per-point embedding, following this up with end-to-end training with a copy of the parent modelâs middle encoder and classifier.

<!-- image-->  
(a) Distance Filtering

<!-- image-->  
(b) Bounding-box Filtering

<!-- image-->  
(c) Voxel Filtering  
Figure 3. Pairwise Gaussian-Point Filtering. (a) Distance filtering only evaluates Gaussian-Point pairs within a radius of a Gaussianâs mean. (b) Bounding-box Filtering evaluates Gaussian-Point pairs when a point falls within the axis-aligned bounding box center on a Gaussian. (c) Voxel Filtering evaluates Gaussian-Point pairs when a point lies in a voxel occupied with sufficiently high likelihood by a given Gaussian.

## 3.5. Filtering via Explicit 3D Geometry

Inspired by the pruning and filtering techniques in 3DGS [5, 7, 8, 22], we introduce Pairwise Gaussian-Point Filtering at inference-time in our encoder to further improve computational efficiency.

Computing the Mahalanobis distance has quadratic complexity in input dimension, making it relatively expensive. However, our experiments also reveal that a sizable percentage of the calculated Gaussian likelihoods are very small (see Fig. 2 in the supplemental). If the likelihood is sufficiently small, we can potentially filter it out and instead assume it to be zero. This requires a heuristic that is significantly faster to compute than Mahalanobis distance. We experiment with three heuristics:

(i) Distance Filtering. We compute the Euclidean distance to each Gaussianâs mean, which only requires linear complexity in dimension as opposed to quadratic. We then threshold the distances by $2 \lambda _ { g } \log { \left( \frac { \alpha _ { g , \operatorname* { m a x } } } { t _ { \mathrm { d i s t a n c e } } } \right) }$ , where $\lambda _ { g }$ is the largest eigenvalue of the covariance matrix. We only evaluate the likelihood for Gaussian-point pairs below this distance. In essence, this method bounds an anistropic Gaussian with an isotropic Gaussian. This is variant of the method used by 3DGS [17].

(ii) Bounding-box Filtering. We compute the minimal axis-aligned bounding box for each Gaussian confidence ellipsoid given a threshold $t _ { \mathrm { b b o x } } ,$ which requires bilinear computational cost in input dimension and number of Gaussians. Then, we check to see if a point falls within a bounding box before computing its likelihood. This is closely related to the âSnug-Boxâ technique proposed by Hanson et al. [14] for 3DGS, except extended to arbitrary dimensions rather than 2D.

(iii) Voxel Filtering. We coarsely voxelize the input volume by $D _ { \mathrm { v o x e l } }$ in each dimension and pre-compute the maximum weighted likelihood of each Gaussian for points falling within each voxel. We cache the list of Gaussians with weighted likelihood above a threshold $t _ { \mathrm { v o x e l } } .$ and at runtime we only compute the likelihood of Gaussian-point pairs for each pointâs voxelâs Gaussian list. This is related to the âAccuTileâ technique proposed by Hanson et al. [14] for 3DGS, except we pre-compute the weighted likelihoods rather than derive them with an interative algorithm.

Each of these methods comes with various advantages and disadvantages. Method (i) benefits from re-using the computation of the difference between the points and the means, but is a poor heuristic if the Gaussians are highly anistropic. Method (ii) requires extra computation to determine bounding box occupancy, but more tightly encloses highly anistropic Gaussians than (i). Finally, method (iii) can be implemented with very low computational costs at runtime using by a lookup table for the Gaussian lists and is the most accurate heuristic given a large enough $D _ { \mathrm { v o x e l } } ,$ but similar to LUTI-MLP [28], has exponential memory requirements in input dimension.

## 3.6. Implementation Details

We implement our encoder in PyTorch, using [37] as a reference PointNet implementation for distillation experiments. Since our 3D Gaussian Point Encoder does not employ a feature transform (only an input transform via T-Net), we modify the PointNet implementation to remove the feature transform. We utilize Mamba3Dâs official release for our Mamba3D experiments, including their released weights for distillation.

When training both our PointNet and Mamba3D variants end to end with natural gradients, we utilize SGD for the Gaussian parameters with a learning rate of 0.005 on the means and 0.005 on the Cholesky decomposition parameters, while using AdamW for the rest of the network. During distillation experiments, we train all components of our models using the AdamW optimizer with a learning rate of $1 . 6 \times 1 0 ^ { - 3 }$ for the means, $5 \times 1 0 ^ { - 4 }$ for the diagonal Cholesky elements, and $1 \times 1 0 ^ { - 4 }$ for the lower triangular Cholesky elements, mixture coefficients, and biases. MLPs used for 3D Gaussian T-Nets and classifiers utilize a learning rate of $1 \times 1 0 ^ { - 4 }$ . The learning rates for the encoder parameters are reduced by a factor of 100 when fine-tuning following the initial distillation.

At test time, we pre-compute the precision matrices to avoid unnecessarily recomputing them for every point cloud. Furthermore, after computing the transformation matrix from the 3D Gaussian T-Net in our PointNet experiments, we apply the inverse transform to the Gaussian parameters rather than apply the transform to the points themselves. This reduces computational costs as there are significantly fewer Gaussians than points per sample.

## 4. Experiments

## 4.1. Shape Classification with PointNet and Mamba3D

We benchmark our encoder on shape classification using the ModelNet40 [36] and ScanObjectNN [31] datasets. Model-Net40 consists of 9,843 training and 2,468 testing meshes of axis-aligned CAD models across 40 classes. We utilize the hardest âPB T50 RSâ variant of ScanObjectNN, consisting of 11,416 training and 2,882 testing real-world 3D scans across 15 object classes. For both datasets, we reserve 25% of the training samples as validation data for our ablations and hyperparameter selection. Following common practice, we report both class-averaged accuracy (mAcc.) and overall accuracy (OA) as our metrics. Furthermore, we measure FLOPs using FVCore. To gauge performance on varying hardware platforms, we measure GPU and CPU latency using PyTorchâs profiler. GPU latency is measured on a single RTX 4070 Mobile GPU with a point cloud size of 2048 points, while CPU latency is measured on a low power ARM CPU (Rockchip RK3588) for methods that do not require custom CUDA extensions.

## 4.1.1. Shape Classification Baselines

For our PointNet experiments, we primarily compare against a PointNet with both an input transform and feature transform, as well as GPointNet, LUTI-MLP, and a PointNet pruned according to [2]. For our Mamba3D experiments, we instead compare against other Transformer and Mamba based architectures. Additionally, we include hierarchical architectures in PointNet++ [26], PointMLP [21], and PointNeXT [27], as well as a near state-of-the-art method in DeLA [3] for reference. All methods are evaluated without voting or cross-modal pre-training. We utilize rotation around the vertical axis for ScanObjectNN, and scaling ${ \tt b y } \pm 2 0 \%$ and translation by Gaussian noise with a standard deviation of 0.01 for ModelNet40. For both Point-Net and Mamba3D experiments, we set $N _ { G }$ to 32 and we utilize the Mahalanobis distance natural gradient. Our filtered PointNet model utilizes bounding-box filtering at test time with $t _ { \mathrm { b b o x } }$ of 0.10.

Table 1. Shape Classification Results. FLOPs and Latency are computed per sample on ScanObjectNN with 2048 input points. X indicates that the model cannot be run on CPU, N indicates end-to-end with natural gradients, D indicates distilled, F indicates filtered. \* indicates weights are not publicly available, so we cannot directly compare memory and latency.
<table><tr><td rowspan="2">Method</td><td colspan="2">ModelNet40</td><td colspan="2">ScanObjectNN</td><td rowspan="2">FLOPs (G)</td><td rowspan="2">Params (M)</td><td rowspan="2">GPU Latency (ms)</td><td rowspan="2">CPU Latency (ms)</td><td rowspan="2">Memory (MB)</td></tr><tr><td></td><td>mAcc. (%) OA (%)</td><td>mAcc. (%)</td><td>OA (%)</td></tr><tr><td colspan="10">PointNet-Like Architectures</td></tr><tr><td>PointNet [25]</td><td>86.1</td><td>90.0</td><td colspan="2">65.2</td><td>0.891</td><td>3.47</td><td>1.00</td><td>110.2</td><td>1057</td></tr><tr><td>PointNet (no FT)</td><td>86.4</td><td>90.2</td><td>65.3</td><td>69.0 69.3</td><td>0.582</td><td>1.61</td><td>0.62</td><td>69.3</td><td>139</td></tr><tr><td>GPointNet [29]</td><td>84.3</td><td>89.2</td><td>58.4</td><td>61.5</td><td>0.052</td><td>1.34</td><td>14.81</td><td>396.7</td><td>2747</td></tr><tr><td>LUTI-MLP [28]</td><td>85.9</td><td>88.0</td><td>60.9</td><td>63.4 71.7</td><td>0.032</td><td>1.03</td><td>3.67</td><td>258.9</td><td>4839</td></tr><tr><td>Pruned PointNet* [2]</td><td>-</td><td>88.2</td><td colspan="2">-</td><td>-</td><td>1.36</td><td>-</td><td>-</td><td>-</td></tr><tr><td>3DGPE (N)</td><td>86.4</td><td>90.1</td><td>65.5</td><td>69.0</td><td>0.068</td><td>1.39</td><td>0.44</td><td>45.7</td><td>573</td></tr><tr><td>PointNet â 3DGPE (D)</td><td>86.1</td><td>90.3</td><td>65.3</td><td>69.1</td><td>0.068</td><td>1.39</td><td>0.44</td><td>45.7</td><td>573</td></tr><tr><td>PointNet â 3DGPE (D + F)</td><td>85.3</td><td>89.8</td><td>65.8</td><td>69.2</td><td>0.064</td><td>1.39</td><td>0.36</td><td>37.6</td><td>605</td></tr><tr><td colspan="10">Dedicated &amp; Hierarchical Architectures</td></tr><tr><td>PointNet++ [26]</td><td>91.8</td><td>89.1</td><td>76.0</td><td>77.8</td><td>1.68</td><td>1.5</td><td>5.9</td><td>403.4</td><td>1215</td></tr><tr><td>PointMLP [21]</td><td>91.3</td><td>94.1</td><td>83.9</td><td>85.4</td><td>31.4</td><td>12.6</td><td>7.7</td><td>xÃÃÃ</td><td>1801</td></tr><tr><td>PointNexXT []</td><td>90.8</td><td>93.2</td><td>85.8</td><td>87.7</td><td>1.6</td><td>1.4</td><td>1.8</td><td></td><td>1257</td></tr><tr><td>DLA 3]</td><td>92.2</td><td>94.0</td><td>89.3</td><td>90.4</td><td>1.5</td><td>5.3</td><td>0.9</td><td></td><td>1177</td></tr><tr><td>Simple View [9]</td><td>-</td><td>93.9</td><td>-</td><td>80.5</td><td>-</td><td>-</td><td>-</td><td></td><td></td></tr><tr><td colspan="10">Transformer and Mamba Architectures</td></tr><tr><td></td><td colspan="10">93.2 - -</td></tr><tr><td>PCT [11] PCM [41]</td><td>90.7</td><td>93.4</td><td>86.6</td><td>- 88.1</td><td>2.3 45.0</td><td>2.9 34.2</td><td>14.8 31.4</td><td>xÃÃx</td><td>6677 5533</td></tr><tr><td>PointMamba [20]</td><td>-</td><td>92.4</td><td>-</td><td>84.9</td><td>3.1</td><td>12.3</td><td>8.3</td><td></td><td>1510</td></tr><tr><td>Mamba3D [13]</td><td>89.7</td><td>93.3</td><td>90.6</td><td>91.6</td><td>3.9</td><td>16.9</td><td>10.4</td><td></td><td>1413</td></tr><tr><td>3DGPE + Mamba3D (N)</td><td>89.9</td><td>93.6</td><td>86.4</td><td>88.0</td><td>1.8</td><td>16.5</td><td>8.2</td><td></td><td>817</td></tr><tr><td>3DGPE + Mamba3D (D)</td><td>89.8</td><td>93.5</td><td>86.6</td><td>88.5</td><td>1.8</td><td>16.5</td><td>8.2</td><td>xÃÃ</td><td>817</td></tr><tr><td>3DGPE + Mamba3D (D + F)</td><td>89.6</td><td>93.3</td><td>86.0</td><td>88.3</td><td>1.8</td><td>16.5</td><td>7.8</td><td></td><td>853</td></tr></table>

## 4.1.2. Comparison to PointNet-like Architectures

All PointNet-style classifiers perform comparably on ModelNet40. However, both GPointNet and LUTI-MLP underperform PointNet on ScanObjectNN compared to Point-Net by approximately 7.8 and 5.9 percentage points respectively. We hypothesize that GPointNetâs relatively low performance arises from its inability to model complex activation volumes, potentially making it harder to deal with the large perturbations present in ScanObjectNN. LUTI-MLPâs lower performance may be also be a result of its modified T-Net, as it uses a tanh activation to constrain point clouds to fit in the unit cube, potentially resulting in deformation that interferes with its interpolation. In comparison, our 3D Gaussian Point Encoder performs comparably to PointNet, achieving the 2nd highest accuracy.

Overall, we find that the 3D Gaussian Point Encoder with PointNet achieves the lowest latency out of all the models tested, achieving approximately 2.7Ã the throughput of a standard PointNet on a mobile GPU and 2.9Ã on a lower power CPU. Interestingly, despite the fact that both GPointNet and LUTI-MLP have lower FLOPs counts, both methods have substantially higher latency. Our latency advantage over these methods also holds on CPU, where both GPointNet and LUTI-MLP become prohibitively expensive, with throughputs under 4 samples per second. In the case of LUTI-MLP, this may be a result of the indexing operations required for interpolating the lookup table only being efficient with custom CUDA kernels.

## 4.1.3. Comparison to Advanced Architectures

Among the Transformer and Mamba architectures, all methods perform comparably on ModelNet40. On ScanObjectNN, Mamba3D performs the best, with our 3D Gaussian Point Encoder performing similarly to PCM. Nonetheless, in comparison to these architectures, our model achieves the lowest FLOPs, latency, and memory. In fact, our model achieves the second lowest memory usage across all model types, highlighting how impactful the encoder design can be towards total memory usage. Compared to Mamba3D, our encoder reduces FLOPs by approximately 54% and memory by 42%, while increasing throughput by 1.27Ã.

## 4.2. Ablations

We ablate the impact of training each of the Gaussian parameters as well as optimization methods. All ablations are carried out on ScanObjectNN with evaluation performed on the validation set. An additional ablation on filtering methods is included in Sec. 2 in the supplemental.

Table 2. Comparisons on $N _ { G }$ and Optimization Methods. Results are class-averaged accuracies on the validation split of ScanObjectNN averaged over 10 trials, listed alongside standard deviation. X denotes incompatibility. Mahalanobis and Fisher refer to the Mahalanobis distance and Fisher information metric natural gradients respectively.
<table><tr><td rowspan="2"> $N _ { G }$ </td><td colspan="5">mAcc. (%)</td></tr><tr><td>Distill</td><td>Mahalanobis</td><td>Fisher</td><td>Adam</td><td>SOAP</td></tr><tr><td colspan="6">3DGPE</td></tr><tr><td>16</td><td> $6 8 . 6 \pm 1 . 0$ </td><td> $6 5 . 9 \pm 5 . 2$ </td><td> $6 6 . 3 \pm 4 . 1$ </td><td> $4 3 . 7 \pm 2 8 . 1 $ </td><td> $6 0 . 0 \pm 6 . 8$ </td></tr><tr><td>24</td><td> $7 1 . 1 \pm 4 . 4$ </td><td> $7 2 . 4 \pm 3 . 6$ </td><td> $6 9 . 9 \pm 5 . 1$ </td><td> $6 5 . 8 \pm 4 . 3$ </td><td> $6 8 . 3 \pm 5 . 3$ </td></tr><tr><td>32</td><td> $8 1 . 6 \pm 4 . 9$ </td><td> $7 8 . 2 \pm 4 . 3$ </td><td> $7 7 . 4 \pm 2 . 4$ </td><td> $5 2 . 4 \pm 2 3 . 6$ </td><td> $7 2 . 9 \pm 6 . 5$ </td></tr><tr><td>64</td><td> $8 2 . 8 \pm 3 . 7$ </td><td> $8 1 . 3 \pm 4 . 2$ </td><td> $7 9 . 5 \pm 3 . 1 $ </td><td> $7 8 . 4 \pm 5 . 0$ </td><td> $7 7 . 4 \pm 7 . 2$ </td></tr><tr><td colspan="6">3DGPE + Mamba3D</td></tr><tr><td>16</td><td> $8 4 . 3 \pm 2 . 4$ </td><td> $8 5 . 1 \pm 2 . 2$ </td><td> $8 2 . 6 \pm 8 . 5$ </td><td> $7 6 . 7 \pm 5 . 3$ </td><td></td></tr><tr><td>24</td><td> $8 5 . 8 \pm 3 . 2$ </td><td> $8 5 . 7 \pm 3 . 7$ </td><td> $8 3 . 8 \pm 4 . 6$ </td><td> $7 7 . 5 \pm 6 . 9$ </td><td></td></tr><tr><td>32</td><td> $8 7 . 6 \pm 2 . 0$ </td><td> $8 6 . 9 \pm 1 . 5$ </td><td> $8 7 . 5 \pm 2 . 1 $ </td><td> $7 8 . 6 \pm 6 . 6$ </td><td>ÃXÃx</td></tr><tr><td>64</td><td> $8 8 . 2 \pm 2 . 1 $ </td><td> $8 7 . 6 \pm 2 . 6$ </td><td> $8 7 . 2 \pm 4 . 2$ </td><td> $8 1 . 4 \pm 7 . 6$ </td><td></td></tr></table>

## 4.2.1. Varying the Number of Gaussians

We report the validation accuracies as we tweak the number of Gaussians, $N _ { G }$ in Tab. 2. Intuitively, performance generally increases as $N _ { G }$ increases, as the Gaussian mixtures are better able to approximate PointNetâs activation volumes. However, the performance does not significantly improve when increasing $N _ { G }$ from 32 to 64.

## 4.2.2. Optimization Techniques

To validate the impact of natural gradients and distillation, we train both of our 3D Gaussian Point Encoder-based models from scratch with Adam [18], and only our PointNet model with SOAP [33], as Mamba3D immediately returns NaN loss with it. In Tab. 2 we demonstrate that training both models end-to-end with Adam necessitates a substantial increase in $N _ { G }$ to achieve acceptable performance, consequently resulting in increased computational cost. Moreover, we observe that, for most values of $N _ { G }$ , models trained end-to-end with standard optimizers exhibit significantly higher variability, and even their best-performing trials consistently underperform compared to trials utilizing either PointNet guidance or natural gradients. We hypothesize that this elevated variance arises from both the limited number of tunable parameters, which makes the optimization process more fragile, and heightened sensitivity to parameter initialization, especially with respect to the Gaussian means. We believe the preconditioned mean updates from both natural gradient methods, and to a lesser extent, SOAP, allow them to mitigate some of this sensitivity.

## 4.2.3. Trainable Parameters

We experiment with fixing the means, lower triangular covariance entries, and diagonal covariance entries of each of the Gaussians in our PointNet experiments. Fixing the lower triangular elements to zero makes the Mahalanobis Distance calculation more efficient but constrains the Gaussians to be axis-aligned, while fixing all the covariance entries constrains all Gaussians to have identity covariances. The results of this experiment are shown in Tab. 3. In general, we find that all three Gaussian parameters contribute strongly to the model performance, with an especially sharp reduction in performance with diagonal covariance.

Table 3. Ablations on Trainable Gaussian Parameters. Results are class-averaged accuracies on the validation split of ScanObjectNN, and are averaged over 5 training runs with distillation. Performance generally goes down as more parameters are fixed.
<table><tr><td rowspan="2"> $N _ { G }$ </td><td colspan="3">Trainable Parameters</td><td rowspan="2">mAcc. (%)</td></tr><tr><td>Mean</td><td>L. Triang.</td><td>Diag.</td></tr><tr><td rowspan="4">16</td><td>â</td><td>â</td><td>â</td><td>68.6</td></tr><tr><td>X</td><td>â</td><td></td><td>64.3 (-4.3)</td></tr><tr><td>X</td><td>X</td><td>V</td><td>59.9 (-8.7)</td></tr><tr><td>X</td><td>X</td><td>X</td><td>60.6 (-8.0)</td></tr><tr><td rowspan="4">32</td><td>â</td><td>â</td><td>V</td><td>81.6</td></tr><tr><td>X</td><td>â</td><td>1</td><td>62.2 (-19.4)</td></tr><tr><td>X</td><td>X</td><td></td><td>66.0 (-15.6)</td></tr><tr><td>X</td><td>X</td><td>X</td><td>65.5 (-16.1)</td></tr><tr><td rowspan="4">64</td><td>â</td><td>â</td><td>â</td><td>82.8</td></tr><tr><td>X</td><td>â</td><td></td><td>75.4 (-7.4)</td></tr><tr><td>X</td><td>X</td><td></td><td>72.2 (-10.6)</td></tr><tr><td>X</td><td>X</td><td>X</td><td>69.7 (-13.1)</td></tr></table>

## 5. Discussion and Conclusion

In this paper, we introduced the 3D Gaussian Point Encoder, a novel point embedding architecture inspired by the explicit geometry of 3D Gaussian Splatting. Our experiments demonstrate that, when trained via natural gradients or 3D Knowledge Distillation, the 3D Gaussian Point Encoder achieves performance comparable to Point-Net while significantly surpassing it in computational efficiency, delivering 2.7Ã higher throughput while using 46% less memory. Furthermore, the encoder integrates well into modern architectures like Mamba3D, improving throughput by 1.27Ã and reducing memory by 42%.

## 5.1. Limitations

Our 3D Gaussian Point Encoder requires more careful optimization techniques than PointNet, and will likely not be suitable in cases where PointNet embeddings do not perform adequately. On Mamba3D experiments, 3DGPE was unable to optimize to the full level of performance as the original model. Furthermore, we only focus on classification. Higher dimensional inputs, such as decorators used in semantic segmentation and detection, may present unexpected challenges in fitting the Gaussian representation.

## References

[1] Shun-ichi Amari. Natural gradient works efficiently in learning. Neural Computation, 10(2):251â276, 1998. 2, 4

[2] Amrijit Biswas, Md Ismail Hossain, MM Elahi, Ali Cheraghian, Fuad Rahman, Nabeel Mohammed, and Shafin Rahman. 3d point cloud network pruning: When some weights do not matter. arXiv preprint arXiv:2408.14601, 2024. 6, 7

[3] Binjie Chen, Yunzhou Xia, Yu Zang, Cheng Wang, and Jonathan Li. Decoupled local aggregation for point cloud learning. arXiv preprint arXiv:2308.16532, 2023. 6, 7

[4] Tri Dao and Albert Gu. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. In International Conference on Machine Learning (ICML), 2024. 2

[5] Stavros Diolatzis, Tobias Zirr, Alexander Kuznetsov, Georgios Kopanas, and Anton Kaplanyan. N-dimensional gaussians for fitting of high dimensional functions. In ACM SIG-GRAPH 2024 Conference Papers, pages 1â11, 2024. 1, 3, 5

[6] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research, 12(7), 2011. 2, 5

[7] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, Zhangyang Wang, et al. Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps. Advances in neural information processing systems, 37: 140138â140158, 2025. 2, 5

[8] Guangchi Fang and Bing Wang. Mini-splatting: Representing scenes with a constrained number of gaussians. In European Conference on Computer Vision, pages 165â181. Springer, 2024. 1, 2, 5

[9] Ankit Goyal, Hei Law, Bowei Liu, Alejandro Newell, and Jia Deng. Revisiting point cloud shape classification with a simple and effective baseline. In International conference on machine learning, pages 3809â3820. PMLR, 2021. 7

[10] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023. 2

[11] Meng-Hao Guo, Jun-Xiong Cai, Zheng-Ning Liu, Tai-Jiang Mu, Ralph R Martin, and Shi-Min Hu. Pct: Point cloud transformer. Computational visual media, 7(2):187â199, 2021. 2, 7

[12] Vineet Gupta, Tomer Koren, and Yoram Singer. Shampoo: Preconditioned stochastic tensor optimization. In International Conference on Machine Learning, pages 1842â1850. PMLR, 2018. 2, 5

[13] Xu Han, Yuan Tang, Zhaoxuan Wang, and Xianzhi Li. Mamba3d: Enhancing local features for 3d point cloud analysis via state space model. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 4995â5004, 2024. 2, 3, 4, 7

[14] Alex Hanson, Allen Tu, Geng Lin, Vasu Singla, Matthias Zwicker, and Tom Goldstein. Speedy-splat: Fast 3d gaussian splatting with sparse pixels and sparse primitives. arXiv preprint arXiv:2412.00578, 2024. 1, 3, 6

[15] Alex Hanson, Allen Tu, Vasu Singla, Mayuka Jayawardhana, Matthias Zwicker, and Tom Goldstein. Pup 3d-gs: Principled uncertainty pruning for 3d gaussian splatting. arXiv preprint arXiv:2406.10219, 2024. 1, 2

[16] Andrew Jones. Natural gradients. https : //andrewcharlesjones.github.io/journal/ natural-gradients.html. Accessed: 2025-09-19. 4

[17] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023. 1, 2, 5

[18] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of the 3rd International Conference on Learning Representations (ICLR), 2015. 2, 5, 8

[19] Alex H Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, and Oscar Beijbom. Pointpillars: Fast encoders for object detection from point clouds. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12697â12705, 2019. 1

[20] Dingkang Liang, Xin Zhou, Wei Xu, Xingkui Zhu, Zhikang Zou, Xiaoqing Ye, Xiao Tan, and Xiang Bai. Pointmamba: A simple state space model for point cloud analysis. Advances in neural information processing systems, 37:32653â32677, 2025. 2, 7

[21] Xu Ma, Can Qin, Haoxuan You, Haoxi Ran, and Yun Fu. Rethinking network design and local geometry in point cloud: A simple residual mlp framework. arXiv preprint arXiv:2202.07123, 2022. 1, 2, 6, 7

[22] Michael Niemeyer, Fabian Manhardt, Marie-Julie Rakotosaona, Michael Oechsle, Daniel Duckworth, Rama Gosula, Keisuke Tateno, John Bates, Dominik Kaeser, and Federico Tombari. Radsplat: Radiance field-informed gaussian splatting for robust real-time rendering with 900+ fps. arXiv preprint arXiv:2403.13806, 2024. 1, 2, 5

[23] Yatian Pang, Eng Hock Francis Tay, Li Yuan, and Zhenghua Chen. Masked autoencoders for 3d point cloud selfsupervised learning. World Scientific Annual Review of Artificial Intelligence, 1:2440001, 2023. 2

[24] Jooyoung Park and Irwin W Sandberg. Approximation and radial-basis-function networks. Neural computation, 5(2): 305â316, 1993. 2

[25] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 652â660, 2017. 2, 3, 7

[26] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. Advances in neural information processing systems, 30, 2017. 1, 6, 7

[27] Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Hammoud, Mohamed Elhoseiny, and Bernard Ghanem. Pointnext: Revisiting pointnet++ with improved training and scaling strategies. Advances in neural information processing systems, 35:23192â23204, 2022. 2, 6, 7

[28] Yusuke Sekikawa and Teppei Suzuki. Tabulated mlp for fast point feature embedding. arXiv preprint arXiv:1912.00790, 2019. 1, 2, 6, 7

[29] Teppei Suzuki, Keisuke Ozawa, and Yusuke Sekikawa. Rethinking pointnet embedding for faster and compact model. In 2020 International Conference on 3D Vision (3DV), pages 791â800, 2020. 1, 2, 3, 7

[30] Linda S L Tan. Analytic natural gradient updates for cholesky factor in gaussian variational approximation. Journal of the Royal Statistical Society Series B: Statistical Methodology, 2025. 5

[31] Mikaela Angelina Uy, Quang-Hieu Pham, Binh-Son Hua, Duc Thanh Nguyen, and Sai-Kit Yeung. Revisiting point cloud classification: A new benchmark dataset and classification model on real-world data. In International Conference on Computer Vision (ICCV), 2019. 2, 6

[32] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Åukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 2

[33] Nikhil Vyas, Depen Morwani, Rosie Zhao, Mujin Kwun, Itai Shapira, David Brandfonbrener, Lucas Janson, and Sham Kakade. Soap: Improving and stabilizing shampoo using adam, 2025. 2, 5, 8

[34] Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, and Hengshuang Zhao. Point transformer v2: Grouped vector attention and partition-based pooling. Advances in Neural Information Processing Systems, 35:33330â33342, 2022. 2

[35] Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. Point transformer v3: Simpler faster stronger. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4840â4851, 2024. 2

[36] Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang, and Jianxiong Xiao. 3d shapenets: A deep representation for volumetric shapes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1912â1920, 2015. 2, 6

[37] Xu Yan. Pointnet/pointnet++ pytorch. https : //github.com/yanx27/Pointnet_Pointnet2_ pytorch, 2019. GitHub repository. 6

[38] Yan Yan, Yuxing Mao, and Bo Li. Second: Sparsely embedded convolutional detection. Sensors, 18(10):3337, 2018. 1

[39] Zhifan Ye, Chenxi Wan, Chaojian Li, Jihoon Hong, Sixu Li, Leshu Li, Yongan Zhang, and Yingyan Celine Lin. 3d gaussian rendering can be sparser: Efficient rendering via learned fragment pruning. Advances in Neural Information Processing Systems, 37:5850â5869, 2025. 1, 2

[40] Xumin Yu, Lulu Tang, Yongming Rao, Tiejun Huang, Jie Zhou, and Jiwen Lu. Point-bert: Pre-training 3d point cloud transformers with masked point modeling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19313â19322, 2022. 2

[41] Tao Zhang, Haobo Yuan, Lu Qi, Jiangning Zhang, Qianyu Zhou, Shunping Ji, Shuicheng Yan, and Xiangtai Li. Point cloud mamba: Point cloud learning via state space model. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 10121â10130, 2025. 2, 7

[42] Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip H.S. Torr, and Vladlen Koltun. Point transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 16259â16268, 2021. 2

[43] Jianqiao Zheng, Xueqian Li, Sameera Ramasinghe, and Simon Lucey. Robust point cloud processing through positional embedding. In 2024 International Conference on 3D Vision (3DV), pages 1403â1412, 2024. 1

[44] Kun Zhou, Zhong Ren, Stephen Lin, Hujun Bao, Baining Guo, and Heung-Yeung Shum. Real-time smoke rendering using compensated ray marching. In ACM SIGGRAPH 2008 papers, pages 1â12. 2008. 2

[45] Yin Zhou and Oncel Tuzel. Voxelnet: End-to-end learning for point cloud based 3d object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4490â4499, 2018. 1