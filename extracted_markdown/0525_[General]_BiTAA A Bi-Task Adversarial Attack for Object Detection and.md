# BiTAA: A Bi-Task Adversarial Attack for Object Detection and Depth Estimation via 3D Gaussian Splatting

Yixun Zhang, Feng Zhou and Jianqin Yin\*

School of Intelligent Engineering and Automation, Beijing University of Posts and Telecommunications, China {zhangyixun, zhoufeng, jqyin}@bupt.edu.cn

Abstractâ Camera-based perception is critical to autonomous driving yet remains vulnerable to task-specific adversarial manipulations in object detection and monocular depth estimation. Most existing 2D/3D attacks are developed in task silos, lack mechanisms to induce controllable depth bias, and offer no standardized protocol to quantify crosstask transfer, leaving the interaction between detection and depth underexplored. We present BiTAA, a bi-task adversarial attack built on 3D Gaussian Splatting that yields a single perturbation capable of simultaneously degrading detection and biasing monocular depth. Specifically, we introduce a dual-model attack framework that supports both full-image and patch settings and is compatible with common detectors and depth estimators, with optional expectation-overtransformation (EOT) for physical reality. In addition, we design a composite loss that couples detection suppression with a signed, magnitude-controlled log-depth bias within regions of interest (ROIs) enabling controllable near or far misperception while maintaining stable optimization across tasks. We also propose a unified evaluation protocol with cross-task transfer metrics and real-world evaluations, showing consistent crosstask degradation and a clear asymmetry between Det â Depth and from Depth â Det transfer. The results highlight practical risks for multi-task camera-only perception and motivate crosstask-aware defenses in autonomous driving scenarios.

## I. INTRODUCTION

Modern autonomous driving systems [1], [2], [3] rely on camera-based perception, where object detection and monocular depth form the front end for downstream tracking, forecasting, and planning. Cameras are attractive for their cost and availability, and the two tasks largely share visual evidence such as textures, edges, and occlusions [4], [5]. As these modules feed into the decision chain, even moderate perception bias can accumulate and compromise safety [6]. This motivates studying adversarial robustness specifically for camera-based detection and depth, and understanding how perturbations propagate across tasks.

Recent adversarial studies on camera-based perception have advanced both digital and physical attacks on detectors, including full-image and patch formulations, white-/blackbox regimes, and the use of expectation-over-transformation (EOT) to improve physical realism and reproducibility [7], [8], [9]. For monocular depth, attacks typically inflate reconstruction errors or distort global scale, with progress from model-specific to model-agnostic settings [10], [11].

Beyond 2D pixels, differentiable-rendering approaches optimize parametric 3D representations (meshes, NeRFs, 3D Gaussians) and bridge to image-space losses, offering crossview consistency and natural integration with physical constraints [12], [13], [14]. These lines of work provide strong baselines and practical toolkits, however, research is still largely organized in task silos, and cross-task effects between detection and depth remain under-studied [15].

To be more specific, detection and depth often share the same camera input and exploit overlapping visual cues. Yet most prior methods optimize for one task at a time, leaving open how to construct a single perturbation that, after multiview rendering, consistently suppresses detection while also biasing monocular depth. The challenge is to coordinate potentially competing objectives within one parameter space and maintain stable gradients across views (Challenge 1). Besides, system-level risks frequently stem from consistent near/far misperception rather than indiscriminate error inflation. Existing depth attacks rarely provide signed and magnitude-controlled bias constrained to semantic regions of interest (ROIs). A practical design must balance physical realizability (e.g., color budgets) with geometric plausibility (e.g., shape stability) to avoid degenerate solutions (Challenge 2). Beyond per-task performance, practitioners need to know whether a perturbation optimized against crosstask transfer and whether the two directions differ. We denote this by DetâDepth as train on detection, evaluate on depth, and Depth â Det as train on depth, evaluate on detection. This motivates a standardized, model-agnostic protocol to quantify both directions under a fixed view set and consistent reporting statistics (Challenge 3).

To address these aforementioned challenges, we propose a novel bi-task adversarial attack built on 3D Gaussian Splatting (BiTAA) aimed at simultaneously achieving adversarial effectiveness in both detection and depth estimation tasks. First, we formulate a unified framework that directly optimizes a single perturbation in the 3D Gaussian parameter space and renders it across views via a fixed differentiable renderer. The task models are kept frozen for compatibility with common detectors and depth estimators, and EOT is optionally applied to the detection branch to model physical variability [8]. All 14 degrees of freedom per Gaussian are updated, and view aggregation promotes cross-view consistency and stable gradients during joint optimization. This turns shared-input coupling into a single, end-to-end optimization problem in a geometry-aware parameterization, addressing Challenge 1. Second, we introduce a composite loss that couples detection suppression with a signed, magnitudecontrolled log-depth bias restricted to per-view ROIs. The direction and amplitude are governed by a sign variable and a target magnitude; ROIs are constructed from detector boxes with inward shrinkage, and clean depth predictions are cached for stable residuals. To ensure physical plausibility and geometric coherence, we add shape-stability penalties on positions, scales and quaternions, together with realizability regularization via a color-space $L _ { \infty }$ budget, total variation on residual images, and an optional high-frequency penalty. This design enables controllable near/far misperception while preserving plausible geometry and stable training dynamics, tackling Challenge 2. Third, we establish a unified crosstask transfer protocol comprising DetâDepth, DepthâDet, and Joint settings. The evaluation reports per-view maximum confidence for detection, ROI-averaged log-depth shifts with sign agreement for depth, and a normalized transferefficiency metric. The protocol uses a fixed view set and consistent statistics, facilitating reproducible comparisons across models and revealing directional asymmetry that informs cross-task-aware defenses, handling Challenge 3.

Our main contributions are as follows:

â¢ We introduce BiTAA, a unified bi-task adversarial framework in a 3D Gaussian parameter space that jointly degrades detection and biases monocular depth after multi-view rendering, with frozen task models and optional EOT for physical variability.

â¢ We propose a composite loss that combines detection suppression with signed, magnitude-controlled logdepth bias within ROIs, together with shape-stability and realizability regularization for physical plausibility and numerical stability.

â¢ We propose a standardized cross-task transfer protocol with model-agnostic metrics for confidence suppression, ROI-averaged depth shifts, and normalized transfer efficiency, revealing clear directional asymmetry. The results of the experiment present the BiTAAâs superior adversarial effectiveness both in detection (15.03% mAP) and depth estimation (0.1989 AbsRel) task.

## II. RELATED WORK

Camera-only perception for autonomous driving. Modern autonomous driving stacks rely on camera-based object detection and monocular depth to support downstream tracking, forecasting, and planning, where these modules share visual evidence such as textures, edges, and occlusions [4], [5]. Cameras remain attractive for their cost and availability, and the robustness of detection-depth perception is therefore central to overall system safety [6].

Adversarial attacks on detection and depth (2D). For camera-based detection, prior work [16], [17], [18], [19] spans digital and physical settings, full-image and patch formulations, and white-/black-box regimes; expectation-overtransformation (EOT) is widely adopted to improve physical reality and evaluation reproducibility [8], [20], [9]. These methods have demonstrated strong effectiveness on standard detectors and provide practical toolkits for evaluating robustness [7]. For monocular depth, attacks typically increase reconstruction error or distort global scale, with steady progress from model-specific to model-agnostic formulations [10], [11]. However, ROI-constrained, signed and magnitudecontrolled log-depth bias remains under-explored, and crosstask transfer is rarely quantified [15].

3D differentiable-rendering based attacks. Beyond 2D pixels, recent approaches [21], [12], [13], [22] optimize parametric 3D scene/object representations (meshes, NeRFs, and 3D Gaussians) and bridge to image-space losses via differentiable rendering. This paradigm offers cross-view consistency, integrates naturally with physical constraints and printing pipelines, and has shown promising lab-to-field behavior [23]. Among these, 3D Gaussian splatting (3DGS) provides an efficient, gradient-friendly space for parameter updates [14]. Most existing 3D attacks, however, primarily target detection, with fewer studies treating detection and monocular depth jointly [24].

Limitations of prior work. While the above lines of research have established powerful baselines and practical methodologies, there is still no unified setting where a single perturbation simultaneously suppresses detection and induces a signed, magnitude-controlled log-depth bias within semantic regions, nor a standardized protocol to measure cross-task transfer between detection and depth [25]. Our study builds on these advances by formulating a bi-task attack in a 3D Gaussian parameter space and adopting a unified transfer protocol to systematically assess such interactions.

## III. METHOD

## A. Preliminaries

3D Gaussian Splatting as a Backbone. We model the target object as a set of N anisotropic 3D Gaussians

$$
\begin{array} { r } { \mathcal { G } = \big \{ ( \mu _ { i } , ~ \alpha _ { i } , ~ s _ { i } , ~ q _ { i } , ~ c _ { i } ) \big \} _ { i = 1 } ^ { N } , } \end{array}\tag{1}
$$

where $\mu _ { i } \in \mathbb { R } ^ { 3 }$ is the position, $\alpha _ { i } \in \mathbb { R } _ { + }$ the opacity, $s _ { i } \in \mathbb { R } ^ { 3 }$ the axis-aligned scale, $q _ { i } \in \mathbb { R } ^ { 4 }$ a unit quaternion (rotation), and $c _ { i } \in [ 0 , \bar { 1 } ] ^ { 3 }$ the RGB color. Each Gaussian thus carries 14 degrees of freedom (DoFs). We treat 3DGS generation as a frozen backbone that, given a few posed images, provides an initial set $\mathcal { G } _ { 0 }$

Differentiable Rendering & Optimization. Given a calibrated view v, a fixed differentiable renderer R produces RGB and alpha from $\mathcal { G } ;$ we render a finite view set V to promote multi-view consistency:

$$
I _ { \nu } = \mathcal { R } ( \mathcal { G } , \nu ) , \quad \nu \in V = \{ \nu _ { 1 } , . . . , \nu _ { m } \} .\tag{2}
$$

Starting from ${ \mathcal { G } } _ { 0 } ,$ we optimize a (possibly masked) subset of the 14D parameters per Gaussian while keeping $\mathcal { R }$ fixed; image-space losses (introduced next) are aggregated over V and back-propagated to G :

$$
\begin{array} { r } { \mathcal { G } = \mathcal { G } _ { 0 } + \Delta \mathcal { G } , \qquad \Delta \mathcal { G } \in \mathbb { R } ^ { N \times 1 4 } . } \end{array}\tag{3}
$$

Unless otherwise stated, all task networks used later are frozen; gradients flow from image-space losses through R to the parameters in $\mathcal { G } .$ The concrete bi-task objectives and their aggregation across views V will be introduced in the next sections.

## B. Overview

BiTAA performs a single perturbation in the 3D Gaussian parameter space so that, after differentiable rendering from multiple viewpoints, the resulting images concurrently (i) suppress detection confidence and (ii) induce a signed, controllable bias in monocular depth within target regions. The 3DGS generation network is treated as a frozen backbone that provides an initial set $\mathcal { G } _ { 0 } ;$ our optimization directly updates G .

Pipeline. As shown in Figure 1, given a view set $V = \{ \nu _ { 1 } , \ldots , \nu _ { m } \}$ , we render $I _ { \nu } = \mathcal { R } ( \mathcal { G } , \nu )$ , optionally apply expectation-over-transformation (EOT) for physical reality on the detection branch, and feed the images to two frozen task models (detector and depth estimator). The overall objective couples a detection-suppression loss with a logdepth bias loss restricted to a per-view ROI $\Omega _ { \nu } ,$ along with regularizers enforcing shape stability and printability.

## C. Bi-Task Adversarial Framework

Forward-backward pipeline. Starting from the backbone output G0, we enable all 14 DoFs of the Gaussian set $\mathcal { G }$ and render each calibrated view $\nu \in V { : }$

$$
I _ { \nu } = \mathcal { R } ( \mathcal { G } , \nu ) , \qquad I _ { \nu } ^ { 0 } = \mathcal { R } ( \mathcal { G } _ { 0 } , \nu ) .\tag{4}
$$

We then feed $I _ { \nu }$ (and optionally its EOT-augmented version $\tau _ { t } ( I _ { \nu } )$ with $t \sim \mathcal { T } )$ to a frozen detector $\mathcal { D }$ and a frozen monocular depth estimator ${ \mathcal { M } } .$ . The detector provides classfiltered boxes and scores for the detection branch; the depth estimator provides dense predictions $d _ { \nu } = \mathcal { M } ( I _ { \nu } )$ , while the clean depth $d _ { \nu } ^ { 0 } = \mathcal { M } ( I _ { \nu } ^ { 0 } )$ is cached once per view for stable residuals. This forms a standard forward pass through the two tasks without training them.

Per-view ROI for depth control. Depth manipulation is confined to semantically relevant regions on each view. Specifically, we construct a per-view ROI $\Omega _ { \nu }$ from the detectorâs boxes after confidence filtering and an inward isotropic shrink (ratio $\rho \in ( 0 , 1 ) ,$ to suppress border noise. Equivalently, we use a binary mask $M _ { \nu }$ with $\Omega _ { \nu } = \{ x \mid$ $M _ { \nu } ( x ) = 1 \}$ . This ROI is only used to localize the depthbias term; detection is still computed on the full image.

View aggregation and optimization. Losses from the detection and depth branches are averaged over the view set V (and over EOT samples if enabled) and combined with shape/printability regularizers; exact definitions are given in Sec. III-D. We update $\mathcal { G }$ by first-order optimization while keeping ${ \mathcal { R } } , { \mathcal { D } }$ and M frozen:

$$
{ \mathcal { G } }  { \mathcal { G } } - \eta \nabla _ { { \mathcal { G } } } { \mathcal { L } } ( { \mathcal { G } } ) .\tag{5}
$$

Two single-task variants are obtained by disabling one task term $( \lambda _ { \mathrm { d e p } } { = } 0 ~ \mathrm { o r } ~ \lambda _ { \mathrm { d e t } } { = } 0 )$ ; they are used later to probe crosstask transfer, while the joint objective reveals interactions between detection and depth.

## D. Composite Loss Function

BiTAA couples a detection-suppression loss with a signed log-depth bias (restricted to per-view ROIs), alongside regularizers for geometric stability and printability. All losses are aggregated over the view set V and back-propagated to the 3D Gaussian parameters.

Detection suppression. For each view $\nu \in V$ and an EOT sample $t \sim \mathcal { T }$ with image operator Ït , let $p _ { \nu } ( t )$ denote the maximum confidence among vehicle-related classes returned by the frozen detector on the augmented image.

$$
p _ { \nu } ( t ) = \operatorname* { m a x } _ { k \in \mathcal { K } _ { \nu } } s _ { \nu , k } ^ { ( t ) } ,\tag{6}
$$

where $s _ { \nu , k } ^ { ( t ) }$ is the confidence score of detection k on $\tau _ { t } ( I _ { \nu } )$ and $\mathcal { H } _ { \nu }$ indexes retained detections after class/score filtering.

$$
\mathcal { L } _ { \mathrm { d e t } } = \frac { 1 } { | V | } \sum _ { \nu \in V } \mathbb { E } _ { t \sim \mathcal { T } } \big [ - \log \big ( 1 - p _ { \nu } ( t ) + \delta \big ) \big ] ,\tag{7}
$$

with a small $\delta > 0$ to avoid numerical issues. This form drives the strongest detection toward zero confidence.

Signed log-depth bias. Let $d _ { \nu }$ and $d _ { \nu } ^ { 0 }$ be the depth maps from the frozen estimator on $I _ { \nu }$ and the clean rendering $I _ { \nu } ^ { 0 } { \mathrm { . } }$ respectively (the latter cached once). Define the per-pixel log-depth residual $\Delta \ell _ { \nu } ( x )$ and average it within the per-view ROI â¦v (skipping views with $\left| \Omega _ { \nu } \right| = 0 )$ .

$$
\Delta \ell _ { \nu } ( x ) = \log \bigl ( d _ { \nu } ( x ) + \varepsilon \bigr ) - \log \bigl ( d _ { \nu } ^ { 0 } ( x ) + \varepsilon \bigr ) ,\tag{8}
$$

$$
\mathcal { L } _ { \mathrm { d e p } } = \frac { 1 } { \left| V \right| } \sum _ { \nu \in V } \ \frac { 1 } { \left| \Omega _ { \nu } \right| } \sum _ { x \in \Omega _ { \nu } } \ \Big ( \Delta \ell _ { \nu } ( x ) \ - \ s \beta \Big ) ^ { 2 } ,\tag{9}
$$

where $\varepsilon > 0$ ensures stability, $s \in \{ + 1 , - 1 \}$ controls the direction (push-far / pull-near), and $\beta > 0$ sets the target bias magnitude.

Geometric stability. To preserve object reality, we penalize deviations of geometry-related parameters from the backbone output $\mathcal { G } _ { 0 } = \{ \overline { { ( } } \mu _ { i } ^ { 0 } , \alpha _ { i } ^ { 0 } , \overline { { } } s _ { i } ^ { 0 } , q _ { i } ^ { 0 } , c _ { i } ^ { 0 } ) \overline  { \} } _ { i = 1 } ^ { N }$ 1. Let $\Delta \mu _ { i } = \mu _ { i } - \mu _ { i } ^ { 0 } , \Delta s _ { i } =$ $s _ { i } - s _ { i } ^ { 0 }$ , and $\Delta q _ { i } = q _ { i } - q _ { i } ^ { 0 }$

$$
\begin{array} { r l } {  { \mathcal { L } _ { \mathrm { s h a p e } } = \sum _ { i = 1 } ^ { N } \Big ( w _ { \mu } \| \Delta \mu _ { i } \| _ { 2 } ^ { 2 } + w _ { s } \| \Delta s _ { i } \| _ { 2 } ^ { 2 } + w _ { q } \| \Delta q _ { i } \| _ { 2 } ^ { 2 } \Big ) } } \\ & { + \zeta \sum _ { i = 1 } ^ { N } ( \| q _ { i } \| _ { 2 } ^ { 2 } - 1 ) ^ { 2 } . } \end{array}\tag{10}
$$

where $\{ w _ { \bullet } \}$ weight per-parameter penalties and the last term softly enforces unit quaternions.

Printability. We constrain perturbations to remain modest in color-space and visually smooth in image-space. Let $\Delta c _ { i } =$ $c _ { i } - c _ { i } ^ { 0 }$ denote per-Gaussian color changes, and $R _ { \nu } = I _ { \nu } - I _ { \nu } ^ { 0 }$ the residual image.

$$
\mathcal { L } _ { \infty } = \operatorname* { m a x } \biggr ( 0 , \underset { i \in [ 1 , N ] } { \operatorname* { m a x } } \underset { c \in \{ r , g , b \} } { \operatorname* { m a x } } | \Delta c _ { i } ^ { ( c ) } | - \varepsilon _ { \infty } \biggr ) ,\tag{11}
$$

$$
\mathcal { L } _ { \mathrm { T V } } = \frac { 1 } { | V | } \sum _ { \nu \in V } \ \left. \nabla R _ { \nu } \right. _ { 1 } ,\tag{12}
$$

$$
\mathcal { L } _ { \mathrm { H F } } = \frac { 1 } { \vert V \vert } \sum _ { \nu \in V } \ \langle W , \vert \mathcal { F } ( R _ { \nu } ) \vert \rangle ,\tag{13}
$$

<!-- image-->  
Fig. 1. BiTAA framework. A frozen 3DGS backbone provides G0, and we optimize all 14 DoFs to obtain G . With optional EOT, multi-view renderings feed (i) a detector, and (ii) a monocular depth head computing a signed log-depth bias within ROIs. Detection suppression, depth-bias, and lightweight regularizers are aggregated over views and back-propagated to update G .

where Îµâ is the color budget, â is the finite-difference gradient (isotropic TV), $\mathcal { F }$ is the DFT magnitude, and W is a ring-shaped mask emphasizing high spatial frequencies. The realizability term combines these components:

$$
{ \mathcal { L } } _ { \mathrm { p r i n t } } = { \mathcal { L } } _ { \infty } + \alpha _ { \mathrm { T V } } { \mathcal { L } } _ { \mathrm { T V } } + \gamma _ { \mathrm { H F } } { \mathcal { L } } _ { \mathrm { H F } } .\tag{14}
$$

Overall objective. The full loss is a weighted sum of the foregoing components, aggregated over views (and EOT for detection):

$$
\mathcal { L } _ { \mathrm { ( } } \mathcal { G } ) = \lambda _ { \mathrm { d e t } } \mathcal { L } _ { \mathrm { d e t } } + \lambda _ { \mathrm { d e p } } \mathcal { L } _ { \mathrm { d e p } } + \lambda _ { \mathrm { s h a p e } } \mathcal { L } _ { \mathrm { s h a p e } } + \lambda _ { \mathrm { p r i n t } } \mathcal { L } _ { \mathrm { p r i n t } } .\tag{15}
$$

We ignore the depth term on views with $\left| \Omega _ { \nu } \right| = 0 ;$ EOT is applied only to the detection branch unless stated. All task networks remain frozen, and gradients flow through the renderer to $\mathcal { G } .$

## E. Cross-Task Transfer Protocol

To quantify how optimizing for one task transfers to the other, we define two single-task training protocols with bitask evaluation, plus a joint baseline. All protocols use the same fixed view set V and frozen task models.

DetâDepth. Train with detection-only loss by disabling the depth term:

$$
\lambda _ { \mathrm { d e t } } > 0 , \qquad \lambda _ { \mathrm { d e p } } = 0 .\tag{16}
$$

Evaluation. Let mAPclean and $\mathrm { m A P ^ { a d v } }$ denote clean and adversarial mAP@0.5 aggregated over V . We report a tasknormalized relative change (TNR) for detection,

$$
\mathrm { T N R _ { d e t } = \frac { \ m A P ^ { c l e a n } - m A P ^ { a d v } } { \ m A P ^ { c l e a n } } \in [ 0 , 1 ] , }\tag{17}
$$

and the cross-task depth change using AbsRel (refer to IV-$\mathbf { A } ) .$

$$
\mathrm { T N R } _ { \mathrm { d e p t h } } = \frac { \mathrm { A b s R e l } ^ { \mathrm { a d v } } - \mathrm { A b s R e l } ^ { \mathrm { c l e a n } } } { \mathrm { A b s R e l } ^ { \mathrm { c l e a n } } } .\tag{18}
$$

DepthâDet. Train with depth-only loss by disabling the detection term:

$$
\lambda _ { \mathrm { d e p } } > 0 , \qquad \lambda _ { \mathrm { d e t } } = 0 .\tag{19}
$$

Evaluation. We again compute $\mathrm { T N R } _ { \mathrm { d e p t h } }$ and the cross-task detection change $\mathrm { T N R } _ { \mathrm { d e t } }$ as above.

## IV. EXPERIMENT

## A. Experimental Setup

Victim models. We evaluate bi-task transfer across three frozen detectors and three frozen monocular depth estimators under the same input resolution and preprocessing. Detectors include Faster R-CNN [26], Mask R-CNN [27], and SSD [28], all pretrained on COCO and run with standard score-thresholding. Depth estimators include, Monodepth2 [29], DPT-Large [30], and Depth Anything V1 [31]. We use the authorsâ released weights and default inference resolutions. All victim models remain frozen throughout attack optimization.

Baseline models. We compare BiTAA with representative attacks on camera-based perception under a unified protocol. Detection baselines include CAMOU [16], UPC [17], DAS [18], FCA [19], and TT3D [21], covering digital and physical formulations commonly used for object detectors. Depth baselines include APA [32], SAAM [33], and 3D2FOOL [34], which directly optimize depth-oriented objectives to distort estimated geometry. For fairness, all methods are run against the same frozen task models and input resolution, with author-recommended hyperparameters when available and minimal tuning on a held-out set; training steps and augmentation options are matched to our setting when applicable.

Attack configuration. We optimize the 3D Gaussian parameterization end-to-end, enabling all 14 degrees of freedom per Gaussian while keeping the renderer and task networks frozen. The composite objective uses fixed scalar weights to balance branches: $\lambda _ { \mathrm { d e t } }$ and $\lambda _ { \mathrm { d e p } }$ for detection and depth, together with $\lambda _ { \mathrm { s h a p e } }$ and $\lambda _ { \mathrm { p r i n t } }$ for regularization; unless otherwise specified, we set $( \lambda _ { \mathrm { d e t } } , \lambda _ { \mathrm { d e p } } , \lambda _ { \mathrm { s h a p e } } , \lambda _ { \mathrm { p r i n t } } ) =$ $( 1 . 0 , 1 . 0 , 0 . 2 , 0 . 1 )$ . To control depth bias, we specify a signed target $( s , \beta )$ , where $s \in \{ + 1 , - 1 \}$ encodes the $\ ' _ { \mathrm { f a r } } \ '$ versus ânearâ bias and $\beta \in [ 0 , 0 . 1 0 ]$ sets the desired magnitude of the ROI-averaged log-depth shift; $\beta = 0$ reduces to detection-only optimization, while larger $\beta$ increases the command strength of bias in a dose-responsive manner as validated in Sec. IV-C.

Camera setup. All renderings use square images at $5 1 2 \times$ 512 with a shared pinhole intrinsic across views. For the physical evaluation, we use a 1:30 Audi A2 model and capture multi-view images with a Redmi K60 Pro smartphone camera at the scale-consistent distance: a real-world 5m test range is emulated by placing the phone approximately 5/30m (â 16.7cm) from the model and sweeping azimuth/elevation to obtain varied viewpoints. The synthetic and physical settings therefore share aligned intrinsics (fixed fovy and identical near/far) and controlled extrinsics, facilitating consistent cross-domain evaluation.

TABLE I  
BASELINE COMPARISON OF BITAA AND EXISTING BASELINE ADVERSARIAL ATTACK METHODS. OUR METHOD BITAA ACHIEVES THE BEST PERFORMANCE ACROSS ALL METRICS, DEMONSTRATING SUPERIOR ADVERSARIAL EFFECTIVENESS AND DEPTH ESTIMATION DEVIATION.
<table><tr><td rowspan="2">Method</td><td>Det. metric</td><td colspan="2">Depth metrics</td></tr><tr><td>mAP (%) â</td><td>AbsRel â</td><td>RMSE â</td></tr><tr><td>Vanilla</td><td>74.68</td><td>0.1878</td><td>1.2906</td></tr><tr><td>CAMOU [16]</td><td>62.15</td><td>0.1881</td><td>1.3002</td></tr><tr><td>UPC [17]</td><td>64.82</td><td>0.1881</td><td>1.3010</td></tr><tr><td>DAS [18]</td><td>49.57</td><td>0.1874</td><td>1.2648</td></tr><tr><td>FCA [19]</td><td>28.24</td><td>0.1887</td><td>1.3022</td></tr><tr><td>TT3D [21]</td><td>29.08</td><td>0.1893</td><td>1.3027</td></tr><tr><td>APA [32]</td><td>58.36</td><td>0.1912</td><td>1.3092</td></tr><tr><td>SAAM [33]</td><td>52.14</td><td>0.1926</td><td>1.3137</td></tr><tr><td>3D2Fool [34]</td><td>67.06</td><td>0.1952</td><td>1.3172</td></tr><tr><td>BiTAA</td><td>15.03</td><td>0.1989</td><td>1.3229</td></tr></table>

Evaluation metrics. For detection error, we report mAP@0.5âmean average precision at an IoU threshold of 0.5âaveraged over the view set V . For depth estimation error, we use (i) Abs Relative difference (AbsRel), defined as $\begin{array} { r } { \frac { 1 } { \left. \Omega \right. } \sum _ { x \in \Omega } \frac { \vert \dot { d } \left( x \right) - d ^ { g t } \left( x \right) \vert } { d ^ { g t } \left( x \right) } } \end{array}$ (â), and (ii) RMSE (log), $\scriptstyle { \sqrt { { \frac { 1 } { | \Omega | } } \sum _ { x \in \Omega } ( d ( x ) - d ^ { g t } ( x ) ) ^ { 2 } } }$ (â) [35], [36]. In addition, to characterize the signed controllability central to our method, we measure the ROI-averaged log-depth displacement âÏ, computed per view as $\begin{array} { r } { \Delta \bar { \sigma _ { \nu } } = \frac { \bar { 1 } } { | \Omega _ { \nu } | } \sum _ { x \in \Omega _ { \nu } } \left[ \bar { \log ( d ( x ) + \varepsilon ) } - \right. } \end{array}$ $\log ( d ^ { 0 } ( x ) + \varepsilon ) ]$ relative to the clean prediction $d ^ { 0 } .$ , and then aggregated across views. In subsequent comparisons with prior methods, we recommend using AbsRel as the primary depth metric (widely adopted and easy to interpret), including RMSE as a secondary indicator in the main table, and reserving âÏ for our cross-task transfer, controllability, and sensitivity analyses where near/far bias and its magnitude are the focus.

## B. Baseline Comparison

Table I compares BiTAA with two families of baselines under a unified protocol (same frozen models, view set, and training budget). Detection-oriented attacks (CAMOU, UPC, DAS, FCA, TT3D) markedly reduce mAP@0.5 relative to the vanilla model (74.68%), with FCA and TT3D reaching 28.24% and 29.08%, respectively, and an overall average of 46.77% across the five methods (i.e., a mean drop of 27.91 points). However, their impact on off-task depth is small or inconsistent: the mean AbsRel over the five methods is 0.18832 (only +0.00052 vs. vanilla 0.1878), and the mean RMSE is 1.29418 (+0.00358 vs. 1.2906), with DAS even decreasing RMSE to 1.2648 (a -0.0258 change). This indicates that detector-only perturbations transfer weakly to depth and may occasionally regularize depth predictions.

<!-- image-->  
Fig. 2. Dose-response of signed log-depth bias.

Depth-oriented attacks (APA, SAAM, 3D2FOOL) produce larger deviations on depth: AbsRel rises to 0.1912/0.1926/0.1952 (average +0.0052 over vanilla), and RMSE to 1.3092/1.3137/1.3172 (average +0.0228), but their effect on detection is moderate, with mAPs of 58.36%, 52.14%, and 67.06% (average 59.19%, a mean drop of 15.49 points). In contrast, BiTAA yields the strongest joint degradation: mAP falls to 6.05%, an absolute reduction of 68.63 points vs. vanilla and a further 22.19 points below the best detector-only baseline (FCA, 28.24%); meanwhile AbsRel reaches 0.1989 (a +0.0111 increase, â¼2.1Ã the depth-only average gain) and RMSE 1.3229 (a +0.0323 increase, â¼1.4Ã the depth-only average gain). Taken together, the results substantiate our design goals: a single 3D perturbation that (i) surpasses detector-only attacks on their primary metric while simultaneously inducing (ii) substantially larger depth deviation than depth-only attacks, evidencing stronger cross-task coupling under the same budget and view setting.

## C. Controllability and Sensitivity

Signed log-depth bias is dose-controllable. We evaluate the controllability promised by our composite loss (Sec. III-D) by sweeping the target magnitude $\beta \in [ 0 , 0 . 1 0 ]$ under both bias directions $s \in \{ + 1 , - 1 \}$ and reporting the ROI-averaged signed log-depth shift âÏ over multiple views/runs. Figure 2 visualizes the empirical means (scatter+line) together with the theoretical targets $\overline { { \Delta \sigma } } = s \cdot \beta$ (slope-Â±1 dashed lines) and their 95% confidence bands estimated from residual scatter. The curves exhibit a clear, nearly linear dose-response: as $\beta$ increases, âÏ grows (or decreases) monotonically with slope close to one and negligible intercept, and the two signed branches (s = +1 for âfarâ, s = â1 for ânearâ) are approximately mirror-symmetric about the origin. The tight confidence bands indicate low variance across views and seeds, evidencing that the optimization remains stable despite multi-view aggregation and ROI masking.

Implications for controllable misperception. These results confirm Contribution 2: our bi-task formulation enables controllable, signed depth bias while jointly optimizing a detection-suppression objective. In practice, a target $s \cdot \beta$ translates to a multiplicative depth change exp(âÏ) within the ROI, allowing us to dial ânearâ or âfarâ misperception with predictable strength. We further observe that the calibration holds across representative models (e.g., DPT-L, DepthAnything, Monodepth2), with model-dependent sensitivity reflected only in the slope tightness rather than the trend itself. Overall, the dose-response behavior validates that a single perturbation in 3DGS space can produce depth shifts that (i) follow the commanded sign and magnitude, and (ii) do so consistently across views, providing a practical handle to probe safety margins in camera-only perception.

TABLE II  
CROSS-TASK TRANSFER: DETâDEPTH.
<table><tr><td rowspan="2">Proxy Det</td><td colspan="3">Target Depth  $\left( \mathrm { T N R } _ { \mathrm { d e p t h } } , \mathrm { \it \% } \right)$ </td><td rowspan="2">Row Mean</td></tr><tr><td>DPT-L</td><td>Mono2</td><td>DA-V1</td></tr><tr><td>Faster R-CNN</td><td>22.82</td><td>9.06</td><td>9.66</td><td>13.85</td></tr><tr><td>Mask R-CNN</td><td>31.44</td><td>19.06</td><td>33.74</td><td>28.08</td></tr><tr><td>SSD</td><td>11.08</td><td>6.62</td><td>10.50</td><td>9.40</td></tr><tr><td>Col Mean</td><td>21.78</td><td>11.58</td><td>17.97</td><td>Overall 17.11</td></tr></table>

TABLE III

CROSS-TASK TRANSFER: DEPTHâDET.
<table><tr><td rowspan="2">Proxy Depth</td><td colspan="3">Target Detector  $\mathrm { ( T N R _ { d e t } , }$  %)</td><td rowspan="2">Row Mean</td></tr><tr><td>FR-CNN</td><td>MR-CNN</td><td>SSD</td></tr><tr><td>DPT-L</td><td>1.49</td><td>2.05</td><td>19.00</td><td>7.51</td></tr><tr><td>Monodepth2</td><td>0.78</td><td>0.88</td><td>3.94</td><td>1.87</td></tr><tr><td>DepthAnything</td><td>7.44</td><td>5.94</td><td>8.86</td><td>7.41</td></tr><tr><td>Col Mean</td><td>3.24</td><td>2.96</td><td>10.60</td><td>Overall 5.60</td></tr></table>

## D. Cross-task Transferability

Cross-task transferability. Under our unified protocol with single-task training and dual-task evaluation, we observe a clear directional asymmetry consistent with our motivation and abstract: attacks optimized for detection transfer strongly to depth, whereas depth-only attacks transfer more weakly to detection. Concretely, the macro-averaged depth degradation in the DetâDepth grid reaches $\overline { { \mathrm { T N R } _ { \mathrm { d e p t h } } } } = 1 7 . 1 \%$ across all models and views, while the macro-averaged detection degradation in the DepthâDet grid is $\overline { { \mathrm { T N R } _ { \mathrm { d e t } } } } = 5 . 6 \%$ . The ratio of these macro means is approximately 3Ã, indicating substantially stronger transfer from detection to depth. These results align with our design of a bi-task framework and composite loss that expose cross-task couplings without changing the task networks, and they validate the claim in the introduction that the interaction between detection and depth is both measurable and asymmetric.

DetâDepth: which targets drive transfer? When using detectors as proxies, two-stage models induce the largest cross-task effect on depth: Mask R-CNN attains the highest row mean (28.08%), followed by Faster R-CNN (13.85%) and SSD (9.40%). On the target side, transformerbased DPT-L is most affected (21.78% column mean), with DepthAnything V1 next (17.97%) and Monodepth2 most resilient (11.58%). We also note salient cells such as Mask R-CNNâDepthAnything at 33.74%. These patterns suggest that detector-driven perturbations alter object-centric cues (e.g., silhouette, shading, and local photometric consistency) that depth estimatorsâespecially global-transformer backbonesâheavily reuse, amplifying cross-task degradation. This observation directly supports our claim that a single 3D perturbation can bias monocular depth while suppressing detection under a standardized evaluation.

TABLE IV  
EFFECT OF REGULARIZERS.
<table><tr><td>Variant</td><td> $\mathrm { T N R } _ { \mathrm { d e t } }$  (%) â</td><td>mAP (%) â</td><td>LPIPS â</td><td>SideÎ (Ã 103) â</td></tr><tr><td>Full (ours)</td><td>69.43</td><td>15.35</td><td>0.5396</td><td>1.7723</td></tr><tr><td>-Lprint only</td><td>70.05</td><td>15.04</td><td>0.5436</td><td>1.7984</td></tr><tr><td>Lshape only</td><td>71.80</td><td>14.16</td><td>0.5612</td><td>2.5537</td></tr><tr><td>â -Both</td><td>73.16</td><td>13.48</td><td>0.5794</td><td>2.7921</td></tr></table>

DepthâDet: where does transfer appear? Using depth estimators as proxies yields smaller but non-negligible transfer to detection. DPT-L and DepthAnything V1 produce comparable row means (7.51% and 7.41%), while Monodepth2 transfers the least (1.87%), consistent with its weaker log-depth bias response in our ablations. As targets, SSD is notably more vulnerable (10.60% column mean) than Faster/Mask R-CNN (3.24% and 2.96%), and the cell DPT-LâSSD peaks at 19.00%. We attribute this to single-stage detectorsâ stronger reliance on dense texture and boundary evidence that is indirectly perturbed by depth-oriented optimization. Overall, while the average DepthâDet transfer is smaller than DetâDepth, the consistent, model-agnostic drops further underscore the practical risk of cross-task coupling in camera-only perception, motivating cross-taskaware defenses and evaluation protocols as advocated in our introduction.

## E. Ablation Study

1) Regularizers: Regularizers trade adversarial strength for realism. Table IV quantifies the impact of the shape and printability terms in our composite loss. We deliberately report only detection-side metrics $\mathrm { ( T N R _ { d e t } }$ and mAP) together with perceptual/parametric realism because depth-side effects are not discriminative here: as established in Sec. IV-C, the signed log-depth bias reliably reaches the commanded target, making depth numbers less informative for ablation. Removing either regularizer yields a modest increase in attack strength (e.g., TNRdet 69.43 â 71.80% and $\mathrm { m A P \ 1 5 . 3 5  1 4 . 1 6 \ f o r \ - \mathcal { L } _ { \mathrm { s h a p e } } ) }$ , and dropping both pushes the strongest detection degradation (TNRdet 73.16%, mAP 13.48). This confirms the intuitive trade-off: the terms do suppress a small portion of adversarial potency, but they are not the primary source of the effect.

Perceptual and parametric stability justify the regularization. In exchange for the small loss of raw attack strength, the regularizers preserve both image-level realism and 3DGS stability. LPIPS (referenced earlier) is lowest for the full model (0.5396) and increases as constraints are removed (0.5794 without both), indicating more noticeable artifacts outside ROIs. To directly measure how much the 3D asset is altered, we report Sideâ, which aggregates average parameter drift across the Gaussian set: Euclidean changes in position and rotation together with mean-squared changes in opacity, scale, and color, summarized as a single scalar $( \times 1 0 ^ { - 3 } ;$ lower is better). Sideâ is smallest with all constraints (1.772), rises mildly without ${ \mathcal { L } } _ { \mathrm { p r i n t } }$ (1.798), and increases markedly without $\mathcal { L } _ { \mathrm { s h a p e } } \left( 2 . 5 5 3 \right)$ and without both (2.792), highlighting the role of the shape term in preventing geometry/appearance drift. Overall, the full setting achieves a favorable balanceâ strong detection degradation with noticeably better realism and parameter stabilityâso we retain both regularizers in all main results.

TABLE V  
EOT IMPROVES ROBUSTNESS UNDER TRANSFORMATIONS.
<table><tr><td>EOT</td><td> $\mathrm { T N R } _ { \mathrm { d e t } }$  â</td><td>mAP (%) â</td><td>LPIPS â</td><td> $\mathrm { V a r } _ { \mathrm { E O T } } ( \Delta \sigma ) \ ( \times 1 0 ^ { 5 } ) \ \downarrow$ </td></tr><tr><td>off</td><td>64.31</td><td>17.92</td><td>0.5837</td><td>1.5256</td></tr><tr><td>partial</td><td>68.04</td><td>16.05</td><td>0.5502</td><td>1.4830</td></tr><tr><td>on</td><td>69.43</td><td>15.35</td><td>0.5396</td><td>1.4324</td></tr></table>

2) EOT Modes.: EOT strengthens transfer under testtime transformations. Table V evaluates attacks on the expected test distribution induced by random photometric/geometric transformations. Training with full EOT yields the strongest detection degradation, with the partial variant in between and off weakest (TNRdet 64.31%; mAP 17.92). This ordering indicates that averaging gradients over transformations does not merely preserve digital strengthâit can improve it when evaluation also involves realistic nuisances. In other words, EOT prevents overfitting to a canonical rendering and aligns the learned 3D perturbation with transformation-stable image evidence, which translates into higher transfer on the transformed test views.

Stability and perceptual quality also benefit from EOT. Beyond raw strength, EOT reduces the dispersion of the commanded log-depth shift across transformations: VarEOT(âÏ) decreases from $1 . 5 2 5 6 \times 1 0 ^ { - 5 }$ (off) to $1 . 4 3 2 4 \times 1 0 ^ { - 5 }$ (on), indicating more consistent bias under the same ROI and target. At the image level, LPIPS monotonically improves $( 0 . 5 8 3 7 ~  ~ 0 . 5 5 0 2 ~  ~ 0 . 5 3 9 6 )$ , suggesting that EOT encourages smoother, less artifact-prone appearance changesâ consistent with the regularization effects observed in Sec. IV. Taken together, these results support the use of EOT when the deployment environment introduces view and rendering variability: it offers stronger cross-task degradation on the relevant test distribution while simultaneously enhancing stability and perceptual realism.

## F. Physical Evaluation

Setup and visualization. Figure 3 composes a 4Ã4 panel for four viewpoints of the 1:30 Audi A2 model, each showing (from left to right) the clean image with detector outputs, its depth heatmap, the adversarial image with detector outputs, and the corresponding adversarial depth heatmap. We fix the detection thresholding/NMS across conditions and annotate the top class and confidence. For depth, the per-row heatmaps share an identical color scale so that cleanâadv comparisons are visually meaningful. Across all views, the clean images are confidently recognized as car (e.g., scores â0.98), whereas the adversarial images display suppressed confidence or misclassification (e.g., umbrella, bicycle, sink)

<!-- image-->  
Fig. 3. Real-world results at four viewpoints. Columns (leftâright): clean image with detection overlays; clean depth heatmap; adversarial image with detection overlays; adversarial depth heatmap. Rows correspond to different azimuth viewpoints; the two depth maps in each row share the same colormap range.

under the same viewpoint, matching the intended detection degradation.

Depth bias in the wild. The adversarial depth maps reveal signed, localized log-depth shifts within the visually salient regions of the vehicle body, while background regions remain largely unaffected. This regional effect is consistent with our training dynamics: although we did not manually constrain where to modify, the 3DGS-based optimization preferentially updates texture-rich body panels (where the ROI has strong image gradients) and leaves the upper body (windows/roof) mostly unchanged. As a result, the induced bias manifests over the car body with the commanded sign, and persists across viewpoints despite lighting changes.

Robustness of the pipeline. Notably, the physical textures used here were produced with a deliberately coarse workflow (generic office printing, simple cutting/adhesion, minor misalignment and surface glare). Even under these non-ideal conditions, we observe consistent detection suppression and depth bias, indicating that the learned perturbation and our training protocol (including EOT) are tolerant to reasonable color reproduction errors, paper/gloss artifacts, and small pose/placement jitters. This aligns with our simulation findings that EOT trades a small amount of digital strength for markedly better stability under real nuisances.

Path to deployment and limitations. A practical transfer path is straightforward: export the optimized perturbation as high-resolution texture segments and fabricate a matte adversarial wrap that can be applied to target surfaces (with fiducial marks for registration, seam-aware tiling, and optional lamination). The minor visual imperfections visible in Fig. 3 (e.g., slight color banding or boundary artifacts) stem from the 3DGS backbone and consumer-grade printing rather than the attack itself; our framework is backboneagnostic and can immediately benefit from future, higherfidelity differentiable 3D reconstructions and better print pipelines. Overall, the results substantiate that a single 3D perturbation realized as a physical texture simultaneously degrades detection and imposes a controllable, signed depth bias in real scenes.

## V. CONCLUSION

We presented BiTAA, a bi-task adversarial attack that operates directly in the 3D Gaussian Splatting (3DGS) parameter space and yields a single perturbation capable of simultaneously suppressing detection and imposing a controllable, signed log-depth bias. Treating 3DGS as a frozen backbone, our dual-model framework couples a detector (with optional EOT for physical realism) and a depth estimator via a composite objective that integrates detection suppression with an ROI-restricted depth-bias term parameterized by $( s , \beta )$ . This design enables controllability (doseâresponse of the commanded bias) while maintaining stability across views and tasks through lightweight shape/printability regularization. We further introduced a unified cross-task transfer protocol and task-normalized metrics, which consistently show cross-task degradation and a clear asymmetry between DetâDepth and DepthâDet transfer. Real-world experiments with printed textures on a scaled model corroborate the digital findings, demonstrating robustness to common physical nuisances.

Looking ahead, our results motivate cross-task-aware defenses for camera-centric autonomy and open avenues to (i) stronger, higher-fidelity differentiable 3D backbones, (ii) broader task sets and multi-sensor perception, (iii) adaptive ROI generation and safety-aware constraints, and (iv) principled, standardized protocols for evaluating multi-task transfer under realistic transformations. We release code and protocols to facilitate reproducible research and to encourage community progress on both attack and defense.

## ACKNOWLEDGMENT

Specific funding information will be provided in the camera-ready version.

## REFERENCES

[1] Autoware.ai, 2020, https://www.autoware.ai/.

[2] B. Apollo, 2020, http://apollo.auto/.

[3] Tesla, 2025, https://www.tesla.com/fsd/.

[4] F. Liu, Z. Lu, and X. Lin, âVision-based environmental perception for autonomous driving,â Proceedings of the Institution of Mechanical Engineers, Part D: Journal of Automobile Engineering, vol. 239, no. 1, pp. 39â69, 2025.

[5] A. Masoumian, H. A. Rashwan, J. Cristiano, M. S. Asif, and D. Puig, âMonocular depth estimation using deep learning: A review,â Sensors, vol. 22, no. 14, 2022.

[6] W.-H. Chen, J.-C. Wu, Y. Davydov, W.-C. Yeh, and Y.-C. Lin, âImpact of perception errors in vision-based detection and tracking pipelines on pedestrian trajectory prediction in autonomous driving systems,â Sensors, vol. 24, no. 15, 2024.

[7] N. Akhtar, A. Mian, N. Kardan, and M. Shah, âAdvances in adversarial attacks and defenses in computer vision: A survey,â IEEE Access, vol. 9, pp. 155 161â155 196, 2021.

[8] A. Athalye, L. Engstrom, A. Ilyas, and K. Kwok, âSynthesizing robust adversarial examples,â 2018.

[9] S. Thys, W. V. Ranst, and T. Goedeme, âFooling automated surveil- Â´ lance cameras: Adversarial patches to attack person detection,â in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2019, pp. 49â55.

[10] Z. Zhang, X. Zhu, Y. Li, X. Chen, and Y. Guo, âAdversarial attacks on monocular depth estimation,â 2020.

[11] A. Wong, S. Cicek, and S. Soatto, âTargeted adversarial perturbations for monocular depth prediction,â in Advances in neural information processing systems, 2020.

[12] C. Xiao, D. Yang, B. Li, J. Deng, and M. Liu, âMeshadv: Adversarial meshes for visual recognition,â in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 6891â 6900.

[13] Y. Fu, Y. Yuan, S. Kundu, S. Wu, S. Zhang, and Y. C. Lin, âNerfool: uncovering the vulnerability of generalizable neural radiance fields against adversarial perturbations,â in Proceedings of the 40th International Conference on Machine Learning (ICML). JMLR.org, 2023.

[14] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3D GaussianÂ¨ Splatting for Real-Time Radiance Field Rendering,â 2023, eprint: 2308.04079.

[15] M. Pham and K. Xiong, âA survey on security attacks and defense techniques for connected and autonomous vehicles,â Computers & Security, vol. 109, p. 102269, 2021.

[16] Y. Zhang, H. Foroosh, P. David, and B. Gong, âCAMOU: Learning Physical Vehicle Camouflages to Adversarially Attack Detectors in the Wild.â International Conference on Learning Representations,International Conference on Learning Representations, Sep. 2018.

[17] L. Huang, C. Gao, Y. Zhou, C. Xie, A. L. Yuille, C. Zou, and N. Liu, âUniversal Physical Camouflage Attacks on Object Detectors,â in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2020.

[18] J. Wang, A. Liu, Z. Yin, S. Liu, S. Tang, and X. Liu, âDual Attention Suppression Attack: Generate Adversarial Camouflage in Physical World,â in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2021.

[19] D. Wang, T. Jiang, J. Sun, W. Zhou, Z. Gong, X. Zhang, W. Yao, and X. Chen, âFCA: Learning a 3D Full-coverage Vehicle Camouflage for Multi-view Physical Adversarial Attack.â Proceedings of the AAAI Conference on Artificial Intelligence, pp. 2414â2422, Jul. 2022.

[20] X. Liu, H. Yang, Z. Liu, L. Song, H. Li, and Y. Chen, âDpatch: An adversarial patch attack on object detectors,â 2019.

[21] Y. Huang, Y. Dong, S. Ruan, X. Yang, H. Su, and X. Wei, âTowards Transferable Targeted 3D Adversarial Attack in the Physical World,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 24 512â24 522.

[22] A. Zeybey, M. Ergezer, and T. Nguyen, âGaussian splatting under attack: Investigating adversarial noise in 3d objects,â in Neurips Safe Generative AI Workshop 2024, 2024.

[23] H. Wei, H. Tang, X. Jia, Z. Wang, H. Yu, Z. Li, S. Satoh, L. Van Gool, and Z. Wang, âPhysical adversarial attack meets computer vision: A decade survey,â IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, pp. 9797â9817, 2024.

[24] J. Zheng, C. Lin, J. Sun, Z. Zhao, Q. Li, and C. Shen, âPhysical 3d adversarial attacks against monocular depth estimation in autonomous driving,â in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 24 452â24 461.

[25] J. Gu, X. Jia, P. de Jorge, W. Yu, X. Liu, A. Ma, Y. Xun, A. Hu, A. Khakzar, Z. Li, X. Cao, and P. Torr, âA survey on transferability of adversarial examples across deep neural networks,â Transactions on Machine Learning Research, vol. 2024, 2024.

[26] S. Ren, K. He, R. Girshick, and J. Sun, âFaster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,â IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 39, no. 6, pp. 1137â1149, 2017.

[27] K. He, G. Gkioxari, P. Dollar, and R. Girshick, âMask R-CNN,â in Â´ 2017 IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2980â2988.

[28] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg, âSSD: Single Shot MultiBox Detector,â in Computer Vision - ECCV 2016. Springer International Publishing, 2016, pp. 21â37, iSSN: 1611-3349.

[29] C. Godard, O. M. Aodha, M. Firman, and G. J. Brostow, âDigging into Self-Supervised Monocular Depth Prediction,â The International Conference on Computer Vision (ICCV), Oct. 2019.

[30] R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, and V. Koltun, âTowards Robust Monocular Depth Estimation: Mixing Datasets for Zeroshot Cross-dataset Transfer,â IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2020.

[31] L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao, âDepth Anything: Unleashing the Power of Large-Scale Unlabeled Data,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2024, pp. 10 371â10 381.

[32] K. Yamanaka, R. Matsumoto, K. Takahashi, and T. Fujii, âAdversarial Patch Attacks on Monocular Depth Estimation Networks,â IEEE Access, vol. 8, pp. 179 094â179 104, 2020.

[33] A. Guesmi, M. A. Hanif, B. Ouni, and M. Shafique, âSAAM: Stealthy Adversarial Attack on Monocular Depth Estimation,â IEEE Access, vol. 12, pp. 13 571â13 585, 2024.

[34] J. Zheng, C. Lin, J. Sun, Z. Zhao, Q. Li, and C. Shen, âPhysical 3D Adversarial Attacks against Monocular Depth Estimation in Autonomous Driving,â in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 24 452â24 461.

[35] D. Eigen, C. Puhrsch, and R. Fergus, âDepth map prediction from a single image using a multi-scale deep network,â in Advances in Neural Information Processing Systems, Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K. Weinberger, Eds., vol. 27. Curran Associates, Inc., 2014.

[36] C. Godard, O. M. Aodha, and G. J. Brostow, âUnsupervised monocular depth estimation with left-right consistency,â in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 6602â 6611.