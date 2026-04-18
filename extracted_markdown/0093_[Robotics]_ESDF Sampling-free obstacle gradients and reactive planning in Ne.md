# Sampling-free obstacle gradients and reactive planning in Neural Radiance Fields

Michael Pantic, Cesar Cadena, Roland Siegwart, and Lionel Ott

{mpantic,cesarc,rsiegwart,lioott}@ethz.ch

Abstractâ This work investigates the use of Neural implicit representations, specifically Neural Radiance Fields (NeRF), for geometrical queries and motion planning. We show that by adding the capacity to infer occupancy in a radius to a pretrained NeRF, we are effectively learning an approximation to an Euclidean Signed Distance Field (ESDF). Using backward differentiation of the augmented network, we obtain an obstacle gradient that is integrated into an obstacle avoidance policy based on the Riemannian Motion Policies (RMP) framework. Thus, our findings allow for very fast sampling-free obstacle avoidance planning in the implicit representation.

## I. INTRODUCTION

In recent years a wealth of novel works on implicit map representations based on deep learning principles were published. Many of these approaches (e.g. [1], [2]) are directly targeted at replacing traditional occupancy mapping systems (such as [3], [4]), and thus are also fed by geometrical data such as pointclouds. Contrastingly, the popular Neural Radiance Fields (NeRFs) [5] are trained on images and optimized for viewpoint synthesis. While a NeRF clearly contains some geometrical information about the scene, the standard loss does not directly enforce the quality or spatial coherence beyond what is needed for rendering images.

In this paper, we investigate the use of NeRFs as an efficient map representation for motion planning, specifically obstacle avoidance which oftentimes leverages obstacle-gradient information. In classical implicit representations the obstacle gradient can be obtained as the derivative of the Signed Distance Function (SDF) [3]. While there has been work on planning with NeRFs [6], these methods employ sampling techniques to approximate the obstacle gradient. This can become a computational bottleneck and might miss intricate features.

This workshop paper contributes novel insights that led to the development of sampling-free reactive collision avoidance policies based on NeRF environment representations. First, we investigate the quality of geometric information present in the layers of a NeRF architecture and augment it to represent an SDF reliably. Second, we show how the NeRFâs backwards differentiation results in an obstacle gradient. We then combine both discoveries with the RMP framework to demonstrate their applicability to obstacle avoidance.

## II. METHOD

In the following we present the architecture and training methods of our architecture.

This work was supported as a part of NCCR Digital Fabrication as well as NCCR Robotics, a National Centre of Competence in Research, funded by the Swiss National Science Foundation (grant number 51NF40 185543).

<!-- image-->  
Fig. 1: Example trajectory followed by an RMP moving from the start location (purple) to the goal (yellow) across the environment. Despite the noisy gradients obtained via the backwards pass, the path is smooth and does not collide with the geometry.

## A. Architecture

A common NeRF consists of three parts â the input positional encoding, 8 fully-connected 128-layers that output a density Ï plus a feature vector, and the color prediction, which uses the feature vector plus a viewpoint angle input to predict color. As color and viewpoint angle are not relevant for geometry, we remove all layers and inputs related to predicting color, and only retain the input encoding and the 8 fully-connected layers of a trained NeRF. The 1D prediction Ï provides a differential probability of density for a specific normalized 3D input coordinate [x, y, z]. To overcome the infinitesimal nature of $\sigma ,$ we add another fully connected layer $l _ { a d d }$ with a single logistic-regression output Î» that is defined as P (obstacle within r) = Î», where r is a normalized radius. To facilitate queries with variable radii, we model r as another input fed into a small intermediate layer $l _ { i r }$ which is then concatenated to the add-on layer $l _ { a d d }$ . Figure 2 visualizes the architecture. All layers are fully connected and use ReLu activation functions.

## B. Attachment layer

We attach the output layer $l _ { a d d }$ after different layers of the NeRF Multi-Layer Perceptron (MLP) during training, effectively truncating the NeRF. Our hypothesis is that information of wider spatial extent is already accessible in the first few layers, as the full layer depth is used to predict an infinitesimal, highly localized Ï density value. By observing differences in training efficiency we hope to gain insight into the depth at which the add-on layer achieves the desired performance while minimizing overall model size.

## C. Training method

We generate queryable occupancy data by regularly sampling $1 2 5 \times 1 0 ^ { 6 }$ Ï-densities from the NeRF and storing points with densities above a threshold in a kd-tree [7]. A training sample is generated by independently sampling x, y, and z coordinates from a uniform distribution $\mathcal { U } ( - 1 , 1 )$ , radius r from a uniform distribution $\mathcal { U } ( 0 . 0 0 5 , 0 . 2 5 )$ , and the âgroundtruthâ occupancy classification, yË, by querying the kd-tree with the sampled $x , y , z , r$ values. We minimize the Binary Cross Entropy (BCE) loss, $\begin{array} { r } { \mathcal { L } = \sum _ { i = 0 } ^ { n } ( - \hat { y } _ { i } \ l o g ( \lambda _ { i } ) + ( 1 - } \end{array}$ $\hat { y } _ { i } ) l o g \big ( 1 - \lambda _ { i } \big ) \big )$ , over occupancy predictions using Adam [8] with a batch size of $n = 1 0 0 0$ for 2500 epochs.

## D. Motion Planning

Riemannian Motion Policies (RMPs) [9] is a framework for motion planning that provides a formulation for combining multiple policies, where each policy consists of a position- and velocity-dependent acceleration $f ( x , { \dot { x } } )$ and a dimensional weighting metric $A ( x , { \dot { x } } )$ . We combine an obstacle avoidance policy with a simple goal attractor policy. For both policies we use the formulation provided in [9]1. The obstacle avoidance policy uses the mapâs information via obstacle gradient âd and distance function d(x).

To obtain âd we simply use the full differentiation of the output w.r.t to the inputs as used in back-propagation, namely $\begin{array} { r } { \nabla d = - \left\lceil \frac { \partial \lambda } { \partial x } , \frac { \partial \lambda } { \partial y } , \frac { \partial \lambda } { \partial z } \right\rceil } \end{array}$ , the obstacle distance d(x) corresponds to the forward query of the network multiplied by â1, see section III-B.

## III. RESULTS

For all experiments we used the Lego dataset2, as it is widely-known and contains adequate geometry for obstacle avoidance planning.

## A. Optimal Attachment Layer

As is visible in Figure 2, the network obtained high accuracy for most attachment depths. Based on the achieved accuracies, we can infer that the add-on layer is able to extract meaningful information from the pre-trained NeRF and that most of that information is present after the first few layers already. For subsequent experiments an attachment depth of 2 is used. While ESDF approximation showed similar results for attachment depth of 1, the obstacle gradient improved by attaching at layer 2.

## B. Occupancy queries and SDF approximation

While the presented architectureâs main goal is to output occupancy probabilities within a certain radius, it can be used to get an approximation to an Euclidian Signed Distance Field (ESDF). Taking the logit outputs directly for a fixed radius parameter and without passing through the sigmoid function, we obtain an approximation to an ESDF. Figure 3 visualizes and compares this approximation against a ground truth ESDF. The resulting ESDF is not metric, but shifted and scaled linearly. Due to the nature of the used obstacle avoidance policy, metric correctness is not needed as we can compensate this with regular parameter tuning as necessary with any RMP.

<!-- image-->

<!-- image-->  
Fig. 2: Left: Proposed architecture, frozen layers of the NeRF are marked in grey. Right: Validation accuracy over the last 100 epochs. Attachment layer is the number of the last used pre-trained NeRF layer.

<!-- image-->  
Fig. 3: Non-sigmoid network output of the proposed architecture (left), ground truth ESDF (middle), and comparison against directly regressing to an ESDF using an MSE loss naively (right). Data normalized to [0, 1]. Mean abs. error of the proposed network is 0.039, of the naive approach 0.175.

## C. Motion planning

Combining all of the aforementioned parts, we obtain a reactive motion planner that can derive the next best acceleration ${ \ddot { x } } = f ( x , { \dot { x } } )$ in continuous space in a single forward $( d ( x ) )$ and backward $( \nabla d ( x ) )$ pass of the network. Figure 1 shows an example path planned using the combination of a goal policy and the described obstacle avoidance policy.

## IV. CONCLUSION

We presented a simple yet effective architecture for gradient-based planning in NeRFs. Our investigations showed that, although trained for image synthesis, the NeRF encapsulates meaningful geometrical information that can be extracted with a simple additional layer. To our surprise most of the NeRF layers are not needed to obtain high accuracy for geometric queries, which shows that introspection and analysis is as import in deep learning as in classical methods. Combining the discoveries of the obtained SDF approximation and obstacle gradient, we demonstrated a sampling-free obstacle avoidance policy that only needs a single forward and backward pass per timestep.

## REFERENCES

[1] J. J. Park, P. Florence, J. Straub, R. Newcombe, and S. Lovegrove, âDeepsdf: Learning continuous signed distance functions for shape representation,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 165â174, 2019.

[2] S. Lionar, L. Schmid, C. Cadena, R. Siegwart, and A. Cramariuc, âNeuralblox: Real-time neural representation fusion for robust volumetric mapping,â in 2021 International Conference on 3D Vision (3DV), pp. 1279â1289, IEEE, 2021.

[3] H. Oleynikova, Z. Taylor, M. Fehr, R. Siegwart, and J. Nieto, âVoxblox: Incremental 3d euclidean signed distance fields for on-board mav planning,â in 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 1366â1373, IEEE, 2017.

[4] A. Hornung, K. M. Wurm, M. Bennewitz, C. Stachniss, and W. Burgard, âOctomap: An efficient probabilistic 3d mapping framework based on octrees,â Autonomous robots, vol. 34, no. 3, pp. 189â206, 2013.

[5] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â in European conference on computer vision, pp. 405â 421, Springer, 2020.

[6] M. Adamkiewicz, T. Chen, A. Caccavale, R. Gardner, P. Culbertson, J. Bohg, and M. Schwager, âVision-only robot navigation in a neural radiance world,â CoRR, vol. abs/2110.00168, 2021.

[7] J. L. Blanco and P. K. Rai, ânanoflann: a C++ header-only fork of FLANN, a library for nearest neighbor (NN) with kd-trees.â https: //github.com/jlblancoc/nanoflann, 2014.

[8] D. P. Kingma and J. Ba, âAdam: A method for stochastic optimization,â arXiv preprint arXiv:1412.6980, 2014.

[9] N. D. Ratliff, J. Issac, D. Kappler, S. Birchfield, and D. Fox, âRiemannian motion policies,â arXiv preprint arXiv:1801.02854, 2018.

<!-- image-->  
Fig. 4: Example of executed plan where the planner was able to avoid intricate obstacles around the cabin of the Lego wheel loader.

<!-- image-->  
Fig. 5: Example of the obstacle gradient obtained by differentiation of the network. Occupancy at $x = 0$ is marked in black.