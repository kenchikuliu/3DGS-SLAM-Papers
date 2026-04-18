# GST-VLA: Structured Gaussian Spatial Tokens for 3D Depth-Aware Vision-Language-Action Models

Md Selim Sarowar1, Omer Tariq2, Sungho Kim\*1

1Yeungnam University 2Korea Advanced Institute of Science and Technology, KAIST

<!-- image-->  
Fig. 1: The proposed GST-VLA pipeline integrates five sequential stages to ground robot actions in structured 3D spatial reasoning. A frozen semantic encoder and a frozen depth expert process the RGB observation in parallel, extracting dense patch features and affine-invariant metric depth respectively. The novel trainable Gaussian Spatial Tokenizer (GST) fuses these streams by back-projecting depth into 3D, estimating per-patch Gaussian parameters $( \mu , \sigma , \alpha )$ from visual features, applying 3D Fourier positional encoding, and aggregating to $N _ { g }$ structured spatial tokens via spatial attention pooling. These tokens are projected into the VLM reasoning core through a cross-attention projector, which generates supervised Depth-Aware Chain-of-Thought (DA-CoT) intermediate reasoning over 3D object grounding, grasp affordance, metric spatial relations, and SE(3) motion plan waypoints before producing action conditioning tokens.

Abstractâ VLA models encode visual observations as 2D patch tokens with no intrinsic geometric structure. Augmenting with dense monocular depth, as in DepthVLA, injects pixeluniform scalar values that encode neither surface orientation nor geometric confidence, and provides no mechanism for intermediate spatial verification before action decoding. We introduce GST-VLA with two contributions. First, the Gaussian Spatial Tokenizer (GST) converts frozen dense depth and frozen semantic patch features into $N _ { g } { = } 1 2 8$ anisotropic 3D Gaussian primitives, each parameterized by a metric residual mean $\boldsymbol { \mu } \in \mathbb { R } ^ { 3 }$ , log-scale covariance $\sigma \in \mathbb { R } ^ { 3 }$ , and learned opacity $\alpha \in ( 0 , 1 )$ . The covariance eigenstructure encodes local surface orientation, and opacity provides per-primitive geometric confidence, both inaccessible from scalar depth. Spatial attention pooling with learned queries concentrates the fixed token budget on geometrically salient regions rather than distributing uniformly. Second, Depth-Aware Chain-of-Thought (DA-CoT) reasoning supervises four structured intermediate spatial thoughts, covering 3D object grounding, grasp affordance contact geometry, pairwise metric distances, and coarse SE(3) waypoints, as explicit generation targets in the training loss. A cross-attention sublayer at every VLM transformer block provides direct access to the raw 256- primitive Gaussian field during DA-CoT generation. A 300Mparameter flow-matching action expert with mixture-of-experts feedforward sublayers decodes 7-DoF delta action chunks via conditional ODE integration, conditioned on both VLM hidden states and DA-CoT outputs through dual cross-attention. Trained with composite $\mathcal { L } _ { \mathrm { f l o w } } + \mathcal { L } _ { \mathrm { C o T } } + \mathcal { L } _ { \mathrm { d e p t h } }$ across three progressive stages, GST-VLA achieves 96.4% on LIBERO (+2.0%), and 80.2% on SimplerEnv (+5.4%). Ablations isolate the contribution of each GST component, each DA-CoT thought, and each training stage, confirming independent and synergistic gains concentrated on precision-demanding tasks.

## I. INTRODUCTION

VLA models [1]â[5] fine-tune large VLMs on demonstration data to produce robot control policies. Visual observations enter these models as 2D patch tokens: a grid of $N _ { p }$ fixedresolution embeddings capturing local appearance statistics. Each patch token occupies a fixed spatial extent in pixel space irrespective of the underlying scene geometry. No patch token encodes depth, surface normal direction, or geometric confidence. When manipulation requires millimeter scale geometric accuracy, such as edge grasping, peg insertion, or thin object picking, the model must recover 3D structure implicitly within its hidden states, a computation that degrades systematically as task precision increases.

DepthVLA [4] addresses this by adding a depth expert as a third transformer stream in a mixture-of-transformers (MoT) design. The depth stream shares attention layers with the VLM and action expert, enabling action tokens to attend to intermediate depth representations rather than a final depth map. This establishes that explicit geometric signals improve manipulation precision. However, three structural limitations remain. (i) The depth representation is pixel uniform: each token holds a scalar depth value at a fixed pixel location, distributing token budget equally across geometrically relevant and irrelevant regions. (ii) No token encodes surface orientation. Scalar depth at a point provides no information about the local tangent plane; a flat surface and a sharp edge at identical depth produce identical representations. (iii) There is no mechanism for the model to explicitly verify or articulate its 3D scene interpretation before generating actions. The spatial reasoning pathway from depth tokens to action tokens is fully implicit and non-inspectable.

GST-VLA addresses all three limitations. The Gaussian Spatial Tokenizer (GST) replaces the dense scalar depth stream with $N _ { g } { = } 1 2 8$ anisotropic 3D Gaussian primitives. Each primitive $\left( \mathbf { c } _ { k } , \Sigma _ { k } , \alpha _ { k } \right)$ is a volumetric spatial token whose axis-aligned covariance $\Sigma _ { k } = \mathrm { d i a g } ( \exp ( 2 \sigma _ { k } ) )$ encodes surface orientation through its eigenstructure (the minimumeigenvalue axis approximates the surface normal), whose opacity $\alpha _ { k } \in ( 0 , 1 )$ encodes geometric confidence (suppressing tokens on specular or textureless surfaces where the depth estimate is unreliable), and whose spatial allocation via learned attention pooling concentrates the fixed token budget on task-relevant geometry. Depth-Aware Chain-of-Thought (DA-CoT) reasoning introduces a supervised intermediate generation stage: the VLM must produce four structured spatial thoughts (3D object centroid, grasp contact geometry, metric spatial relations, SE(3) waypoints) as explicit supervised targets before generating action-conditioning tokens. During this generation, a cross-attention sublayer at every VLM transformer block provides unfiltered access to the full 256-primitive raw Gaussian field, allowing thought-level queries into specific geometric regions.

Our contributions are summarized as follows:

â¢ GST-VLA architecture: The GST, a trainable module producing structured 3D Gaussian tokens from frozen depth and visual features via back-projection, per-patch parameter estimation with multi-scale opacity gating, 3D Fourier positional encoding, and spatial attention pooling.A three-stage training protocol with composite $\mathcal { L } _ { \mathrm { f l o w } } + \mathcal { L } _ { \mathrm { C o T } } + \mathcal { L } _ { \mathrm { d e p t h } }$ objective

â¢ DA-COT: DA-CoT, a supervised intermediate reasoning stage imposing explicit 3D geometric targets as sequential generation within the VLM.

â¢ Data Efficient validation: We demonstrate that GST-VLA with lower computation cost & parameters, significantly outperforms state-of-the-art VLAs in simulated environments (LIBERO, Simpler), achieving notable gains in grasping accuracy, collision avoidance, and overall task success.

## II. RELATED WORK

## A. Spatial Representations in VLA Models

DepthVLA [4] introduces a depth expert as a dedicated transformer stream within a MoT architecture, pretrained on metric depth datasets and jointly trained with the VLM and action expert. The action expert attends to intermediate depth features at every transformer layer, not merely a final depth map. SpatialVLA [3] injects 3D egocentric position encodings derived from an off-the-shelf depth estimator into patch tokens, modulating the 2D features with scalar position offsets, and introduces adaptive spatial action grids for discrete action tokenization. Both approaches share a structural property: depth enters the model as a pixel-uniform scalar, with one value per spatial location. The GST departs from this by parameterizing each spatial token as a full anisotropic Gaussian primitive with seven learned parameters (three for mean offset, three for log-scale covariance, one for opacity), and compressing $N _ { p } { = } 2 5 6$ raw tokens into $N _ { g } { = } 1 2 8$ via learned spatial attention pooling rather than uniform spatial binning. This produces tokens that encode surface orientation and geometric confidence, and that concentrate representational capacity on task relevant regions.

## B. 3D Gaussian Primitives Beyond Rendering

3D Gaussian Splatting [6] represents scenes as sets of anisotropic Gaussian primitives $\left\{ \left( \mu _ { k } , \Sigma _ { k } , \alpha _ { k } \right) \right\}$ optimized via differentiable rendering for novel view synthesis. GaussTR [7] aligns Gaussian representations with foundation model features for self-supervised 3D understanding. GPSToken [8] uses 2D Gaussian functions for spatially-adaptive image tokenization, decoupling spatial layout from texture for generation. The GST departs from all of these in objective: it does not optimize for rendering, understanding, or image tokenization, but uses the Gaussian primitive as a spatial token format consumed by a VLM for manipulation policy conditioning. The Gaussian parameterization is valued not for its rendering properties but for the geometric information it encodes per token: position, anisotropic extent, and confidence, all calibrated against metric depth through a differentiable rendering auxiliary loss.

## C. Chain-of-Thought for Embodied Reasoning

Chain-of-thought reasoning [2] supervises intermediate steps in LLMs. ECoT [9] studies spatial reasoning chains in manipulation settings. CogACT [10] decouples cognition from action by conditioning a diffusion transformer on a cognition token output by the VLM. HybridVLA [11] unifies autoregressive and diffusion action generation within a single LLM but does not introduce structured spatial reasoning targets. None of these works supervise intermediate generation with explicit metric 3D coordinates, grasp contact geometry, or SE(3) waypoints derived from demonstration data. DA-CoT introduces exactly this: four structured thought components, each supervised against offline 3D annotations, each generating tokens that flow through the VLMâs autoregressive decoding before action-conditioning tokens are produced.

## III. METHOD

GST-VLA processes an RGB observation $o _ { t } \in \mathbb { R } ^ { H \times W \times 3 }$ $\scriptstyle ( H = W = 2 2 4 )$ , a language instruction â, and proprioceptive state $s _ { t } \in \mathbb { R } ^ { 7 }$ through five sequential stages. A frozen visual encoder produces dense semantic patch features $\mathbf { F } _ { \mathrm { s e m } } \in$ R256Ã1152. A frozen monocular depth estimator produces affine invariant metric depth $\hat { D } \in \overline { { \mathbb { R } } } ^ { H \times W }$ . Both encoders remain frozen throughout all training stages. The two streams are fused exclusively within the GST.

## A. Gaussian Spatial Tokenizer

The GST is a trainable module that converts the pair $( \mathbf { F } _ { \mathrm { s e m } } , \hat { D } )$ into $N _ { g }$ structured volumetric spatial tokens. Four sequential operations construct the representation.

1) Depth Back-Projection to Metric 3D Anchors: Given metric depth $\hat { D }$ and calibrated camera intrinsics $\ b { K } \in \mathbb { R } ^ { 3 \times 3 }$ each pixel (u, v) is lifted to camera frame 3D via

$$
\mathbf { p } _ { u \nu } = \hat { D } _ { u \nu } \cdot K ^ { - 1 } [ u , \nu , 1 ] ^ { \top } \in \mathbb { R } ^ { 3 } .\tag{1}
$$

For each of the $N _ { p } = 2 5 6$ semantic patches (corresponding to a $1 6 \times 1 6$ pixel receptive field at input resolution $2 2 4 \times 2 2 4 )$ we compute the mean of back-projected 3D coordinates within the receptive field, yielding metric anchors $\{ \mathbf { p } _ { k } \} _ { k = 1 } ^ { N _ { p } }$ . Each anchor $\mathbf { p } _ { k }$ localizes patch k in the camera-frame 3D coordinate system. The anchor precision is bounded by two factors: the depth estimator accuracy (median relative error ${ \sim } 3 \%$ on indoor scenes) and the spatial averaging within the $1 6 \times 1 6$ receptive field, which smooths sub-patch depth variation. The residual offset $\mu _ { k }$ estimated in the next step compensates for this averaging artifact.

2) Per-Patch Gaussian Parameter Estimation: A 4-layer MLP $f _ { \theta }$ with hidden dimensions [1152,768,512,7] and GELU activations maps each semantic patch embedding to a 7-dimensional Gaussian parameterization:

$$
[ \mu _ { k } , \sigma _ { k } , \hat { \alpha } _ { k } ] = f _ { \theta } ( \mathbf { F } _ { \mathrm { s e m } , k } ) ,\tag{2}
$$

decomposed as follows.

The residual mean $\boldsymbol { \mu _ { k } } \in \mathbb { R } ^ { 3 }$ is an offset from the backprojected anchor, yielding the primitive centroid $\mathbf { c } _ { k } = \mathbf { p } _ { k } +$ $\mu _ { k }$ . This residual formulation is structurally important: it decouples the coarse metric localization (provided by the depth estimator through $\mathbf { p } _ { k } )$ from fine geometric refinement (learned by the MLP through $\mu _ { k } )$ . Ablating $\mu _ { k }$ by fixing it to zero pins centroids to back-projected anchors and costs 1.9 percentage points (Table IV), confirming that subpatch refinement contributes meaningfully to downstream task performance.

The log-scale $\sigma _ { k } \in \mathbb { R } ^ { 3 }$ parameterizes an axis-aligned anisotropic covariance $\Sigma _ { k } = \mathrm { d i a g } ( \exp ( 2 \sigma _ { k } ) )$ ). The three eigenvalues of $\Sigma _ { k }$ encode the spatial extent of the primitive along each camera-frame axis. For primitives on flat surfaces, the eigenvalue corresponding to the surface normal direction is small (tight extent perpendicular to the surface), while the two tangential eigenvalues are large (diffuse extent along the surface). For primitives on edges or corners, multiple eigenvalues contract. This geometric information is absent from scalar depth representations: two surface regions at identical depth but with different local curvature produce identical depth tokens in DepthVLA but distinct covariance structures in the GST. Replacing anisotropic $\Sigma _ { k }$ with isotropic covariance $( \sigma _ { k } ^ { x } = \sigma _ { k } ^ { y } = \sigma _ { k } ^ { z } )$ costs 1.6 percentage points (Table VII), quantifying the contribution of orientation encoding.

The pre-activation opacity logit $\hat { \alpha } _ { k } \in \mathbb { R }$ is processed through a multi-scale pathway. Rather than computing opacity from the patch feature alone, we aggregate across three spatial scales via a multi-scale image pyramid (MIP):

$$
\alpha _ { k } = \sigma \big ( f _ { \mathrm { e x p } } \big ( \mathrm { M I P } ( \mathbf { F } _ { \mathrm { s e m } , k } ) \big ) \big ) \in ( 0 , 1 ) ,\tag{3}
$$

where $\mathrm { { M I P } ( \cdot ) }$ concatenates the patch feature with its $2 \times$ and $4 \times$ average-pooled spatial neighborhoods, producing a $3 \times 1 1 5 2 = 3 4 5 6 .$ -dimensional input to a 2-layer opacity MLP $f _ { \exp } : \mathbb { R } ^ { 3 4 5 6 } \to \mathbb { R } ^ { 1 }$ . The multi-scale context is necessary because geometric confidence at a patch location depends on surrounding texture gradient magnitude: a patch on a uniform surface surrounded by other uniform patches (low texture gradient at all scales) should receive low opacity because the depth estimate there lacks photometric verification. A patch on a textured object surrounded by background (high gradient locally, low globally) should receive high opacity. The MIP aggregation provides exactly this scale-dependent context. Ablating opacity by fixing $\alpha _ { k } = 1$ costs 1.5 percentage points (Table IV), confirming the value of confidence based token weighting.

3) 3D Fourier Positional Encoding: The metric centroid $\mathbf { c } _ { k }$ is encoded via multi octave 3D sinusoidal features:

$$
\mathrm { P P E } ( \mathbf { c } _ { k } ) = \left[ \sin ( 2 ^ { l } \pi \mathbf { c } _ { k } ) , \cos ( 2 ^ { l } \pi \mathbf { c } _ { k } ) \right] _ { l = 0 } ^ { L - 1 } \in \mathbb { R } ^ { 6 L } ,\tag{4}
$$

with $L = 6$ octaves yielding a 36-dimensional positional code. The choice of 3D Fourier encoding over learned 2D positional embeddings is motivated by a specific requirement: the VLM must be able to compute approximate metric distances between tokens by operating on their positional codes. Two tokens at centroids $\mathbf { c } _ { i }$ and $\mathbf { c } _ { j }$ separated by distance d produce Fourier features whose inner product structure encodes d across multiple frequency bands. Learned 2D positional embeddings encode pixel-space proximity, which conflates depth variation with lateral displacement: two objects at the same pixel column but different depths receive similar 2D positions but require distinct 3D treatment. The 3D Fourier encoding resolves this conflation. Replacing 3D Fourier PE with learned 2D PE costs 2.8 percentage points (Table IV), the largest single-component ablation within the GST.

The per-primitive spatial token is formed by projecting the concatenation $[ \mathbf { F } _ { \mathrm { s e m } , k } ; \mathrm { P P E } ( \mathbf { c } _ { k } ) ; \sigma _ { k } ; \alpha _ { k } ] \in \bar { \mathbb { R } } ^ { 1 1 \bar { 9 } 2 }$ (where $1 1 9 2 = 1 1 5 2 + 3 6 + 3 + 1 )$ through a learned linear projection $\mathbf { W _ { \mathrm { t o k } } } \in \mathbb { R } ^ { 1 1 9 2 \times d _ { g } }$ with $d _ { g } = 7 6 8$ to produce the raw spatial token for patch k.

4) Spatial Attention Pooling: The $N _ { p } = 2 5 6$ raw spatial tokens are compressed to $N _ { g } = 1 2 8$ output tokens via single-

layer cross-attention:

$$
\mathbf { Z } _ { \mathrm { s p a t i a l } } = \operatorname { s o f t m a x } \left( \frac { \mathbf { Q } _ { \mathrm { p o o l } } \mathbf { K } ^ { \top } } { \sqrt { d _ { g } } } \right) \mathbf { V } \in \mathbb { R } ^ { N _ { g } \times d _ { g } } ,\tag{5}
$$

where $\mathbf { Q } _ { \mathrm { p o o l } } \in \mathbb { R } ^ { N _ { g } \times d _ { g } }$ are $N _ { g }$ learned pooling queries, $\mathbf { K } , \mathbf { V } \in$ $\mathbb { R } ^ { N _ { p } \times d _ { g } }$ are key and value projections of the raw token set $\mathbf { T } _ { \mathrm { r a w } }$ . Each learned query specializes to attend to a specific geometric pattern in the raw token set. Queries that correspond to object surfaces attend to dense clusters of similar depth, high opacity, anisotropic primitives. Queries that correspond to background attend to scattered, high-Ï , low-Î± primitives. The resulting allocation is adaptive: a scene with many small objects distributes pooling queries across multiple object surfaces; a scene with one large object and empty background concentrates queries on the object and assigns few to background. This contrasts with the fixed uniform allocation of pixel-aligned depth, where token budget is spent equally on the task-relevant cup handle and the far wall. Replacing spatial attention pooling with uniform average pooling costs 2.1 percentage points (Table IV).

The full raw token set $\mathbf { T } _ { \mathrm { r a w } } \in \mathbb { R } ^ { N _ { p } \times d _ { g } }$ is retained separately and used as the geometric reference for DA-CoT crossattention (Section III-B).

5) Differentiable Depth Rendering as Geometric Regularizer: The predicted Gaussian field must remain geometrically consistent with the metric depth target. We enforce this through a differentiable rendering loss. For camera ray r with direction d and origin o, rendered depth is

$$
\hat { D } _ { \mathrm { r e n d e r } } ( { \bf r } ) = \sum _ { k = 1 } ^ { N _ { p } } w _ { k } \cdot ( { \bf c } _ { k } ^ { \top } { \bf d } ) ,\tag{6}
$$

where $w _ { k } = \alpha _ { k } ^ { ( \mathbf { r } ) } \Pi _ { j < k } ( 1 - \alpha _ { j } ^ { ( \mathbf { r } ) } )$ and the ray specific opacity is $\begin{array} { r } { \alpha _ { k } ^ { ( \mathbf { r } ) } = \alpha _ { k } \exp \left( - \frac { 1 } { 2 } \frac { ( t _ { k } - \mathbf { c } _ { k } ^ { \top } \mathbf { d } ) ^ { 2 } } { \mathbf { d } ^ { \top } \Sigma _ { k } \mathbf { d } } \right) } \end{array}$ with $t _ { k } = \| \mathbf { c } _ { k } - \mathbf { o } \|$ . Equation (6) is fully differentiable with respect to $\mu _ { k } , \sigma _ { k }$ , and $\alpha _ { k } ,$ meaning that the depth reconstruction loss ${ \mathcal { L } } _ { \mathrm { d e p t h } }$ provides gradients that geometrically calibrate all GST parameters. This loss acts as a geometric regularizer: without it, the MLP could produce Gaussian parameters that are useful for downstream action prediction (via ${ \mathcal { L } } _ { \mathrm { f l o w } } )$ but geometrically inconsistent with the actual scene, e.g. placing centroids at incorrect depths while compensating through covariance. The depth rendering loss prevents this degenerate solution.

## B. VLM Reasoning Core and DA-CoT

1) Spatial Token Injection: The pooled spatial tokens $\mathbf { Z } _ { \mathrm { s p a t i a l } } \mathbf { \bar { \Lambda } } \in \mathbb { R } ^ { N _ { g } \times d _ { g } }$ are projected into the VLM hidden dimension $d _ { \mathrm { V L M } }$ via a two-layer cross-attention projector Î¨ with residual connection:

$$
\begin{array} { r } { \tilde { \mathbf { Z } } = \Psi ( \mathbf { Z } _ { \mathrm { s p a t i a l } } ) = \mathbf { C } \mathbf { A } \big ( \mathbf { Z } _ { \mathrm { s p a t i a l } } , \mathbf { F } _ { \mathrm { s e m } } \big ) \mathbf { W } _ { \mathrm { p r o j } } , } \end{array}\tag{7}
$$

where $\mathbf { W } _ { \mathrm { p r o j } } \in \mathbb { R } ^ { d _ { g } \times d _ { \mathrm { V L M } } }$ . The cross-attention in Eq. (7) regrounds each spatial token in the full semantic feature space before VLM ingestion: a spatial token representing a cup handle re-attends to the semantic patch features that encode âcupâ appearance, fusing geometric localization with object identity. The VLM input sequence is $\mathcal { X } = [ \tilde { \mathbf { Z } } ; \mathbf { L } _ { \mathrm { t o k e n s } } ; \mathbf { e } _ { s } ]$ (spatial tokens, language embeddings, proprioceptive embedding). The VLM is adapted via LoRA (r=16, Î±=32) on all self-attention projection matrices.

2) Depth-Aware Chain-of-Thought: Standard VLAs collapse 3D scene interpretation and action generation into a single undifferentiated computation within the VLM hidden states. There is no explicit representation of the modelâs 3D understanding that can be supervised, inspected, or verified. DA-CoT introduces a supervised intermediate generation stage that separates these two computations.

The separation is implemented architecturally. During DA-CoT token generation, an additional cross-attention sublayer is inserted at every VLM transformer block:

$$
\mathbf { h } _ { i } ^ { ( \ell ) } \gets \mathbf { h } _ { i } ^ { ( \ell ) } + \mathrm { C A } \Big ( \mathbf { h } _ { i } ^ { ( \ell ) } , \mathbf { T } _ { \mathrm { r a w } } , \mathbf { T } _ { \mathrm { r a w } } \Big ) ,\tag{8}
$$

where $\mathbf { h } _ { i } ^ { ( \ell ) }$ is the hidden state of token i at VLM layer $\ell ,$ and $\dot { \mathbf { T } } _ { \mathrm { r a w } } \in \mathbb { R } ^ { N _ { p } \times d _ { g } }$ is the full raw Gaussian token set (not the pooled version $\mathbf { Z } _ { \mathrm { s p a t i a l } } )$ . The use of raw rather than pooled tokens in Eq. (8) is a deliberate design choice: spatial attention pooling compresses $N _ { p } { = } 2 5 6$ tokens into $N _ { g } { = } 1 2 8$ for efficient VLM processing, but during CoT generation the model needs to query specific geometric regions at full resolution. For instance, generating the grasp contact point c2 requires attending to the specific subset of primitives covering the objectâs graspable surface, which may correspond to only 3-5 raw tokens whose information is diluted by pooling. The cross-attention in Eq. (8) provides this targeted geometric access.

The DA-CoT output is a structured chain $\mathcal { C } = ( c _ { 1 } , c _ { 2 } , c _ { 3 } , c _ { 4 } )$ generated autoregressively before action-conditioning tokens. The four components are ordered by causal dependency: each subsequent thought depends on information produced by preceding thoughts.

$c _ { 1 } { : }$ 3D object grounding. The model generates the metric centroid of the task relevant object in camera coordinates, $e . g .$ âtarget centroid: $( 0 . 1 5 , - 0 . 0 8 , 0 . 4 2 ) ~ \mathrm { m } . ^ { , }$ This requires the VLM to fuse the language instruction (identifying which object) with the Gaussian field (localizing it in 3D). The centroid estimate anchors all subsequent spatial reasoning; errors here propagate through $c _ { 2 } , \ c _ { 3 } ,$ , and $c _ { 4 }$ . Ablating c1 costs 1.9 percentage points (Table V).

c2: Grasp affordance. The model generates a 3D contact point offset relative to the c1 centroid and the approach surface normal direction. This specifies where and at what angle the gripper should engage the target. The contact point requires attending to the specific primitives on the objectâs graspable surface, querying their covariance eigenstructure to infer local surface orientation for the approach vector.

c3: Metric spatial relations. The model generates signed metric distances between task-relevant objects and surfaces in camera frame. For a "place cup on shelf" task, c3 might produce the vertical distance from the cup centroid to the shelf surface and the lateral distance to the shelf edge. These distances condition the action expertâs trajectory height and lateral offset.

c4: SE(3) motion plan. The model generates a coarse sequence of 6-DoF end-effector waypoints (pre-grasp, grasp, post-grasp retract) as $\left[ \Delta x , \Delta y , \Delta z , \Delta r _ { x } , \Delta r _ { y } , \Delta r _ { z } \right]$ deltas in camera frame. These waypoints provide the action expert with a geometric prior over trajectory shape. The action expert refines this coarse plan through flow matching, but the prior substantially constrains the search space. Ablating $c _ { 4 }$ has the largest individual effect among the four components at â2.3 percentage points (Table V).

The full VLM output is $\left( \mathcal { C } , \mathbf { L } _ { \mathrm { a c t i o n } } \right)$ , where $\mathbf { L } _ { \mathrm { a c t i o n } } \in$ $\mathbb { R } ^ { N _ { a } \times d _ { \mathrm { V L M } } }$ are action-conditioning tokens generated after the CoT chain. The sequential ordering ensures that action tokens are generated in a context that includes the explicit 3D reasoning outputs, allowing the VLM to condition action generation on verified spatial understanding.

## C. Flow-Matching Action Expert

The action expert is a 300M-parameter transformer with 6 layers at hidden dimension $d _ { e } = 5 1 2$ . Each layer contains a self-attention block, two cross-attention blocks (one attending to VLM hidden states $\mathbf { H } _ { \mathrm { v l m } } .$ , one attending to DA-CoT action tokens $\mathbf { L } _ { \mathrm { a c t i o n } } )$ , and a mixture-of-experts (MoE) feedforward block with 8 experts per layer, top-2 routing via a learned gating network $g : \bar { \mathbb { R } } ^ { 5 1 2 } \overset { - } { \to } \mathbb { R } ^ { 8 }$ , and expert hidden dimension 2048 with SiLU activation. The dual cross-attention conditioning is:

$$
\mathbf { H } _ { \mathrm { a c t } } ^ { ( \ell ) } = \mathbf { H } _ { \mathrm { a c t } } ^ { ( \ell ) } + \mathrm { C A } \Big ( \mathbf { H } _ { \mathrm { a c t } } ^ { ( \ell ) } , \mathbf { H } _ { \mathrm { v l m } } \Big ) + \mathrm { C A } \Big ( \mathbf { H } _ { \mathrm { a c t } } ^ { ( \ell ) } , \mathbf { L } _ { \mathrm { a c t i o n } } \Big ) ,\tag{9}
$$

where ${ \bf H } _ { \mathrm { a c t } } ^ { ( 0 ) } = { \bf e } _ { s }$ s (proprioceptive embedding). The two conditioning streams are functionally distinct: $\mathbf { H } _ { \mathrm { v l m } }$ carries the VLMâs fused semantic-visual representation; $\mathbf { L } _ { \mathrm { a c t i o n } }$ carries the 3D geometric reasoning distilled by DA-CoT. Removing the $\mathbf { L } _ { \mathrm { a c t i o n } }$ stream costs 3.1 percentage points (Table VI), confirming that DA-CoT outputs encode geometric information that is not redundant with what $\mathbf { H } _ { \mathrm { v l m } }$ already captures.

The action distribution is modeled via conditional flow matching [?]. Given straight-line interpolation ${ \bf a } _ { t } = ( 1 - t ) { \bf a } _ { 0 } +$ ta1 with $\mathbf { a } _ { 0 } \sim \mathcal { N } ( \mathbf { 0 , I } )$ , the velocity field $\nu _ { \theta } : \mathbb { R } ^ { 7 L _ { \mathrm { a c t } } } \times [ 0 , 1 ] \to$ $\mathbb { R } ^ { 7 L _ { \mathrm { { a c t } } } }$ is trained by:

$$
\mathcal { L } _ { \mathrm { f l o w } } = \mathbb { E } _ { t , \mathbf { a } _ { 0 } , \mathbf { a } _ { 1 } } \left[ \left\| \nu _ { \theta } ( \mathbf { a } _ { t } , t \mid \mathbf { H } _ { \mathrm { a c t } } ) - ( \mathbf { a } _ { 1 } - \mathbf { a } _ { 0 } ) \right\| ^ { 2 } \right] .\tag{10}
$$

At inference, the ODE $d \mathbf { a } _ { t } / d t = \nu _ { \theta } ( \mathbf { a } _ { t } , t \mid \mathbf { H } _ { \mathrm { a c t } } )$ is integrated from t=0 to t=1 with 10 Euler steps. The model predicts action chunks of $L _ { \mathrm { a c t } } = 1 0$ future 7-DoF delta poses (6-DoF end effector plus 1-DoF gripper), with temporal ensemble weighting $\delta _ { t } = 0 . 0 1$ s across overlapping chunk predictions.

The MoE feedforward structure enables expert specialization by action phase. Different MoE experts activate during precision-reach versus grasp closure versus retract sub-trajectories, as routed by the combined semantic and geometric conditioning. A single dense feedforward network underfits this multi-modal action distribution, costing 1.7 percentage points (Table VI).

## D. Composite Training Objective

The composite loss is:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { f l o w } } + \lambda _ { \mathrm { C o T } } \mathcal { L } _ { \mathrm { C o T } } + \lambda _ { \mathrm { d e p t h } } \mathcal { L } _ { \mathrm { d e p t h } } ,\tag{11}
$$

with $\lambda _ { \mathrm { C o T } } = 0 . 5$ and $\lambda _ { \mathrm { d e p t h } } = 0 . 1$

LCoT is the token-level cross-entropy for DA-CoT generation:

$$
\mathcal { L } _ { \mathrm { C o T } } = - \sum _ { j = 1 } ^ { 4 } \sum _ { t } \log p _ { \theta } ( y _ { t } ^ { ( j ) } \mid y _ { < t } ^ { ( j ) } , \mathcal { X } ) ,\tag{12}
$$

where $y ^ { ( j ) }$ are ground truth token sequences for thought $c _ { j } .$ This loss provides two gradient pathways. First, gradients through the VLM parameters improve the quality of spatial reasoning and coordinate generation. Second, because the VLM attends to the raw Gaussian tokens during CoT generation (Eq. (8)), gradients from ${ \mathcal { L } } _ { \mathrm { C o T } }$ flow backward through the cross-attention into $\mathbf { T } _ { \mathrm { r a w } }$ and hence into the GST parameters. This means the CoT loss acts as an indirect geometric supervisor for the Gaussian field: if the Gaussian tokens place a primitive at an incorrect 3D location, the VLM will generate an incorrect centroid in $c _ { 1 }$ , producing a large ${ \mathcal { L } } _ { \mathrm { C o T } } .$ , whose gradients adjust the GST parameters to correct the primitive placement. This coupling between ${ \mathcal { L } } _ { \mathrm { C o T } }$ and the GST is a key reason why DA-CoT and GST provide synergistic rather than merely additive gains.

${ \mathcal { L } } _ { \mathrm { d e p t h } }$ is the scale-invariant logarithmic loss between the depth rendered from the Gaussian field (Eq. (6)) and the target metric depth:

$$
\mathcal { L } _ { \mathrm { d e p t h } } = \frac { 1 } { n } \sum _ { i } d _ { i } ^ { 2 } - \frac { 0 . 8 5 } { n ^ { 2 } } \left( \sum _ { i } d _ { i } \right) ^ { 2 } ,\tag{13}
$$

where $d _ { i } = \log \hat { D } _ { \mathrm { r e n d e r } , i } - \log D _ { \mathrm { t a r g e t } , i } .$

## E. Three Stage Training Protocol

The three stage protocol is structured to address a specific ordering constraint: the GST must produce geometrically calibrated tokens before the VLM can learn to reason over them, and the VLM must produce meaningful CoT and action tokens before the full system can be jointly refined.

Stage 1 (S1): GST and action expert pretraining. The GST and action expert are initialized randomly and trained with both encoders and the VLM frozen. ${ \mathcal { L } } _ { \mathrm { d e p t h } }$ supervision uses ScanNet [12], Hypersim [13], and ARKitScenes [14] (diverse indoor scenes with metric depth ground truth). A small robot demonstration split provides ${ \mathcal { L } } _ { \mathrm { f l o w } }$ supervision without CoT. This stage establishes that the Gaussian field is geometrically calibrated: centroids are metrically accurate, covariances reflect surface geometry, and opacities suppress unreliable regions. Without this pretraining, the VLM in S2 receives randomly parameterized Gaussian tokens and cannot learn meaningful spatial reasoning, costing 6.2 percentage points (Table VI). Duration: 80K steps, LR $3 \times 1 0 ^ { - 4 }$ , batch 256, 8ÃA100-80GB.

Stage 2 (S2): LoRA adaptation with DA-CoT supervision. LoRA adapters $( r = 1 6 , \alpha = 3 2 )$ are introduced on the crossattention projector Î¨ and all VLM self-attention projections.

<!-- image-->  
Fig. 2: GST-VLA framework. Two frozen encoders produce semantic features and metric depth. The LoRA adapted GST lifts these into $N _ { g } { = } 1 2 8$ anisotropic 3D Gaussian tokens via four operations. A cross-attention projector injects spatial tokens into the VLM, where DA-CoT sublayers attend to the raw 256-primitive Gaussian field. The action expert receives dual conditioning: VLM hidden states (semantic and visual context) and DA-CoT action tokens (3D geometric reasoning).

The action expert is fully fine-tuned. All three loss terms are active. DA-CoT ground truth annotations are generated offline: 3D centroids from open vocabulary detection on depthprojected point clouds; grasp contact points from a pretrained grasp planner; metric distances from 3D bounding boxes; SE(3) waypoints from demonstration end-effector trajectories via velocity zero crossings (threshold 0.5 cm/s) at pre-grasp, grasp-close, and retract phases. Annotation throughput: â¼0.3 s per frame. Duration: 40K steps, LR $1 \times 1 0 ^ { - 4 }$ , batch 128.

Stage 3 (S3): Full fine-tuning. All non-frozen parameters are jointly refined at reduced learning rate. This stage is necessary for cross-modal alignment: the GSTâs geometric representation, the VLMâs CoT generation, and the action expertâs conditioning must be jointly optimized so that improvements in one module propagate correctly to the others. Stopping at S2 costs 2.1 percentage points (Table VI). Duration: 20K steps, LR $3 \times 1 0 ^ { - 5 }$ , batch 64.

## IV. EXPERIMENTS

## A. Setup

Evaluations span three benchmarks. SimplerEnv [15] (visual-matching simulation on BridgeData V2; we report task progress) tests generalization under visual domain shift. LIBERO [16] (130 tasks across Spatial, Object, Goal, Long suites; we report average success rate) tests structured manipulation. The LIBERO-Pro [17] across 6 categories: pickand-place, stacking, drawer manipulation, precision insertion, thin object grasping, and cluttered scenes.

Baselines: OpenVLA [1], CogACT [10], SpatialVLA [3], Ï0VLA [4]. Two ablation variants isolate individual contributions: $\mathrm { G S T - V L A ^ { \dagger } }$ (GST tokens, no DA-CoT) and GST-VLAâ¡ (DA-CoT with plain depth tokens instead of Gaussian tokens).

## B. Main Results

Table I reports data-efficient with lower parameters manipulation success rates. GST-VLA achieves 83.1% overall versus 76.8% for SpatialVLA (+6.3 pp), 68.7% for CogACT (+14.4 pp), 64.3% for Ï0 (+18.8 pp), and 52.3% for OpenVLA (+30.8 pp). The performance gain is non-uniform across categories, which is informative. Precision insertion and thin object grasping show the largest gains over DepthVLA (+9.2 pp and +8.3 pp respectively), precisely the tasks where surface orientation information (encoded in $\Sigma _ { k } )$ and SE(3) waypoint priors (from c4) contribute most directly: inserting a peg requires sub-centimeter alignment where the anisotropic covariance resolves the socketâs angular tolerance; grasping a thin object requires the grasp contact normal (from c2) to be near-parallel to the objectâs flat face. Pick-and-place shows a narrower +2.0 pp gain, consistent with this taskâs weaker dependency on geometric precision.

The ablation variants provide causal evidence. GST-VLAâ  (79.2%) confirms that structured Gaussian tokens alone improve over DepthVLAâs dense depth (+2.4 pp) even without DA-CoT. GST-VLAâ¡ (76.3%) shows that DA-CoT without structured Gaussian tokens provides limited benefit: the plain depth tokens cannot support the fine-grained geometric queries required during CoT generation. The full model exceeds the sum of individual gains from GST-only and CoT-only, confirming synergy.

Table II reports LIBERO results. GST-VLA achieves

TABLE I: Data Efficient Manipulation results (success rate %).
<table><tr><td>Method</td><td>P&amp;P</td><td>Stack</td><td>Drawer</td><td>Insert</td><td>Thin</td><td>Clutter</td><td>Avg.</td></tr><tr><td>OpenVLA</td><td>72.0</td><td>58.0</td><td>53.0</td><td>41.0</td><td>38.0</td><td>52.0</td><td>52.3</td></tr><tr><td>CogACT</td><td>83.0</td><td>74.0</td><td>70.0</td><td>60.0</td><td>57.0</td><td>68.0</td><td>68.7</td></tr><tr><td>SpatialVLA</td><td>88.0</td><td>80.0</td><td>78.0</td><td>71.0</td><td>69.0</td><td>75.0</td><td>76.8</td></tr><tr><td>GST-VLA</td><td>88.0</td><td>83.0</td><td>80.0</td><td>74.0</td><td>72.0</td><td>78.0</td><td>79.2</td></tr><tr><td>GST-VLAâ¡</td><td>87.0</td><td>78.0</td><td>76.0</td><td>74.5</td><td>68.0</td><td>74.0</td><td>76.3</td></tr><tr><td>GST-VLA</td><td>90.0</td><td>85.0</td><td>84.0</td><td>80.2</td><td>77.3</td><td>81.9</td><td>83.1</td></tr></table>

TABLE II: LIBERO benchmark results (average task success rate %).
<table><tr><td>Method</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>OpenVLA</td><td>93.8</td><td>93.0</td><td>90.2</td><td>83.0</td><td>90.0</td></tr><tr><td>$\Ïot0</td><td>95.1</td><td>94.3</td><td>92.8</td><td>87.9</td><td>92.5</td></tr><tr><td>CogACT</td><td>96.0</td><td>95.1</td><td>93.5</td><td>88.3</td><td>93.2</td></tr><tr><td>SpatialVLA</td><td>97.1</td><td>95.8</td><td>95.1</td><td>89.5</td><td>94.4</td></tr><tr><td>GST-VLA</td><td>98.2</td><td>97.4</td><td>97.1</td><td>92.6</td><td>96.4</td></tr></table>

96.4% average, with the largest per-suite gain on LIBERO-Long (+3.1 pp vs. DepthVLA). Long horizon tasks involve sequential manipulations where geometric context must be maintained across sub-tasks; the SE(3) waypoint supervision in $c _ { 4 }$ provides trajectory level geometric coherence that is particularly beneficial for multi-step sequencing. LIBERO-Spatial and LIBERO-Object show moderate gains (+1.1 pp, +1.6 pp), consistent with those suites relying more on object recognition than geometric precision.

Table III reports SimplerEnv results. GST-VLA achieves 80.2% average task progress versus 74.8% for DepthVLA (+5.4 pp). Close-Drawer shows the largest gain (+5.9 pp), where gripper-to-handle alignment benefits from the grasp contact geometry in $c _ { 2 } .$ . An important observation: Gaussian tokens defined in 3D metric coordinates are decoupled from pixel-space appearance. When SimplerEnv applies visual domain shift (background and illumination changes), pixel aligned depth tokens undergo distributional shift through the depth encoderâs sensitivity to appearance, while Gaussian tokens in metric 3D remain more stable because backprojection normalizes the coordinate system to camera frame meters.

## C. Ablation Studies

1) GST Component Ablations: Table IV isolates each GST component by replacing it while keeping all other components at their full configuration.

The 3D Fourier PE ablation (â2.8 pp) is the most informative. Replacing metric 3D Fourier encoding with learned 2D positional embeddings removes the ability to compute approximate metric distances between tokens from their positional features. The VLM can still reason about relative image plane positions, but conflates depth variation with lateral displacement: two objects at the same pixel column but 20 cm apart in depth receive similar positional features. This degrades precisely the tasks where depth discrimination matters.

TABLE III: SimplerEnv benchmark results (task progress %).
<table><tr><td>Method</td><td>Pick-Can</td><td>Move-Near</td><td>Draw.-O</td><td>Draw.-C</td><td>Avg.</td></tr><tr><td>OpenVLA</td><td>62.4</td><td>57.1</td><td>54.3</td><td>63.2</td><td>59.3</td></tr><tr><td> $\pi _ { 0 }$ </td><td>70.1</td><td>61.3</td><td>59.4</td><td>67.4</td><td>64.6</td></tr><tr><td>CogACT</td><td>73.8</td><td>65.2</td><td>63.0</td><td>72.5</td><td>68.6</td></tr><tr><td>SpatialVLA</td><td>78.2</td><td>72.4</td><td>69.8</td><td>78.8</td><td>74.8</td></tr><tr><td>GST-VLA</td><td>83.1</td><td>77.5</td><td>75.6</td><td>84.7</td><td>80.2</td></tr></table>

TABLE IV: GST component ablations. Average success rate (%).
<table><tr><td>Configuration</td><td>Avg.</td><td>â</td></tr><tr><td>3D Fourier PE â 2D learned PE</td><td>80.3</td><td>-2.8</td></tr><tr><td>Attn. pooling â avg. pool</td><td>81.0</td><td>-2.1</td></tr><tr><td> $\alpha _ { k } \equiv 1 { \mathrm { ~ ( n o ~ o p a c i t y ) } }$ </td><td>81.6</td><td>â1.5</td></tr><tr><td> $\mu _ { k } \equiv \mathbf { 0 } { \mathrm { ~ ( n o ~ r e s i d u a l ) } }$ </td><td>81.2</td><td>â1.9</td></tr><tr><td> $N _ { g } = 6 4$ </td><td>79.8</td><td>-3.3</td></tr><tr><td> $N _ { g } = 2 5 6$ </td><td>83.5</td><td>+0.4</td></tr><tr><td> $\mathrm { G S T - V L A } \ ( N _ { g } = 1 2 8 )$ </td><td>83.1</td><td>â</td></tr></table>

The spatial attention pooling ablation (â2.1 pp) reveals the cost of uniform token allocation. Average pooling assigns equal weight to every raw token regardless of opacity or semantic relevance. A single pooled token may aggregate a high-confidence object surface primitive with a low confidence background primitive, diluting the geometric signal.

The $N _ { g }$ sweep reveals a saturation effect: reducing from 128 to 64 tokens costs 3.3 pp because the token budget becomes insufficient to represent all task relevant geometry; increasing from 128 to 256 yields only +0.4 pp at double the computational cost, indicating diminishing marginal return.

2) DA-CoT Component Ablations: Table V removes individual thought components from the CoT supervision.

The SE(3) motion plan $c _ { 4 }$ ablation (â2.3 pp) has the largest individual effect because the waypoint prior directly constrains the action expertâs trajectory shape through $\mathbf { L } _ { \mathrm { a c t i o n } } .$ Without $c _ { 4 } .$ , the action expert must infer all trajectory geometry from the VLMâs visual-semantic hidden states, which encode geometry only implicitly.

The 3D grounding $c _ { 1 }$ ablation (â1.9 pp) is second-largest because $c _ { 1 }$ anchors all subsequent reasoning: errors in the centroid estimate propagate through $c _ { 2 }$ (contact point is relative to centroid), $c _ { 3 }$ (distances are between centroids), and $c _ { 4 }$ (waypoints are relative to the target). The causal dependency structure means that $c _ { 1 }$ quality is a bottleneck for the entire CoT chain.

Removing all DA-CoT $( \mathcal { L } _ { \mathrm { C o T } } = 0 )$ costs 3.9 pp, matching GST-VLAâ  in Table I.

3) Training Protocol and Expert Conditioning: Table VI validates the staged training protocol and dual conditioning.

The S1 ablation (â6.2 pp) is the largest single ablation across all tables. This confirms the ordering constraint: the GST must produce geometrically calibrated Gaussian tokens before the VLM can learn to reason over them.

TABLE V: DA-CoT component ablations. Average success rate (%).
<table><tr><td>Configuration</td><td> $\operatorname { A v g } .$ </td><td>â</td></tr><tr><td rowspan="3">No DA-CoT  $( \mathcal { L } _ { \mathrm { C o T } } = 0 )$  w/o 3D grounding (c1) w/o grasp affordance (c2)</td><td>79.2</td><td>-3.9</td></tr><tr><td>81.2</td><td>-1.9</td></tr><tr><td>81.5 82.0</td><td>-1.6</td></tr><tr><td colspan="2">w/o spatial relations  $\left( c _ { 3 } \right)$  w/o SE(3) motion plan  $( c _ { 4 } )$  80.8 Full DA-CoT  $( c _ { 1 } \mathrm { - } c _ { 4 } )$  83.1</td><td>â1.1 â2.3</td></tr></table>

TABLE VI: Training and conditioning ablations. Average success rate(%).
<table><tr><td>Configuration</td><td> $\operatorname { A v g } .$ </td><td>â</td></tr><tr><td> ${ \bf S } 2 + { \bf S } 3$  only (no S1 pretraining)</td><td>76.9</td><td>-6.2</td></tr><tr><td>S1+S2 only (no S3 end-to-end) No</td><td>81.0</td><td>â2.1</td></tr><tr><td> $\mathbf { L } _ { \mathrm { a c t i o n } }$  conditioning Dense FFN (no MoE) in expert</td><td>80.0</td><td>-3.1</td></tr><tr><td>GST-VLA (S1+S2+S3)</td><td>81.4 83.1</td><td>-1.7</td></tr></table>

Without geometric pretraining, the VLM receives random Gaussian parameterizations and cannot learn meaningful spatial correspondences during S2.

Removing $\mathbf { L } _ { \mathrm { a c t i o n } }$ conditioning (â3.1 pp) confirms that DA-CoT encodes geometric information not redundant with $\mathbf { H } _ { \mathrm { v l m } }$ . This is expected: the VLM hidden states encode geometry implicitly within a representation primarily shaped by language modeling objectives, while $\mathbf { L } _ { \mathrm { a c t i o n } }$ carries explicit 3D reasoning outputs generated by the DA-CoT pathway.

4) Gaussian Tokens vs. Alternative 3D Representations: Table VII provides a controlled comparison at fixed token budget $N _ { g } { = } 1 2 8$ . The hierarchy of results is informative. Dense depth scalars (DepthVLA style binning into 128 regions) lose 4.5 pp: no orientation, no confidence, no adaptive pooling. Surface normal tokens lose 3.0 pp: orientation without metric position or confidence. Point cloud tokens (position only) lose 2.4 pp: the gap between point cloud and full Gaussian (2.4 pp) quantifies the combined contribution of anisotropic covariance and learned opacity. Isotropic Gaussians lose 1.6 pp: this isolates the contribution of orientation encoding. Uniform opacity loses 1.5 pp: this isolates the contribution of confidence weighting. The full Gaussian parameterization, combining metric position, anisotropic orientation, and learned confidence, achieves the best performance across all tested alternatives.

## D. Analysis

1) DA-CoT Reasoning Accuracy: On 200 held out demonstrations with 3D ground-truth annotations, GST-VLA achieves median 3D centroid localization error of 2.3 cm (c1), grasp contact point error of 1.8 cm (c2), and SE(3) waypoint position error of 3.1 cm (c4). The Pearson correlation between composite CoT accuracy and task success rate is 0.71, confirming that DA-CoT quality is a reliable predictor of downstream action quality. This correlation also suggests that monitoring c1 accuracy at deployment time could serve as a runtime confidence metric for execution reliability without requiring ground truth action labels.

TABLE VII: Gaussian tokens vs. alternative representations at $N _ { g } { = } 1 2 8$ . avg (%).
<table><tr><td>Representation</td><td> $\operatorname { A v g } .$ </td><td>â</td></tr><tr><td>Dense depth scalars (DepthVLA-style)</td><td>78.6</td><td>-4.5</td></tr><tr><td>Surface normal tokens</td><td>80.1</td><td>-3.0</td></tr><tr><td>Point cloud tokens (position only)</td><td>80.7</td><td>â2.4</td></tr><tr><td>Gaussian w/o anisotropy (isotropic)</td><td>81.5</td><td>-1.6</td></tr><tr><td>Gaussian w/o opacity  $( \alpha _ { k } \equiv 1 )$ </td><td>81.6</td><td>-1.5</td></tr><tr><td>Full Gaussian tokens</td><td>83.1</td><td></td></tr></table>

2) Gaussian Token Spatial Distribution: Visualizing the $N _ { g } { = } 1 2 8$ pooled tokens as oriented ellipsoids confirms the adaptive allocation pattern predicted by the spatial attention pooling mechanism. Object surfaces and grasp relevant edges attract 60-70% of the pooling queries, yielding dense clusters of small-Ï high-Î± tokens. Background and table surfaces receive 20-30% of queries as diffuse high-Ï low-Î± primitives. The remaining 5-15% of queries receive near zero attention weights and produce effectively null tokens. The opacity field $\{ \alpha _ { k } \}$ reaches values below 0.05 on specular metallic surfaces and textureless white walls, suppressing their contribution to both the differentiable depth rendering and downstream CoT reasoning.

3) Inference Latency: Full pipeline inference runs at 6.2 Hz on a single A100-80GB: encoding 18 ms, GST tokenization 12 ms, DA-CoT generation 38 ms (â¼80 tokens across four thoughts), flow-matching ODE 22 ms. DepthVLA runs at 8.1 Hz and OpenVLA at 9.4 Hz. The 72 ms overhead relative to DepthVLA stems primarily from DA-CoT generation (38 ms) and is acceptable for the 6.2 Hz control frequency used across all experiments.

4) Failure Cases: Performance degrades on highly reflective surfaces (specular metals, glass) and heavily occluded targets. The root cause in both cases is unreliable metric depth from the frozen estimator, which produces inaccurate back-projected anchors $\mathbf { p } _ { u \nu } .$ . The opacity mechanism partially mitigates this by suppressing affected primitives $( \alpha _ { k } < 0 . 0 5$ on specular surfaces), but when the target object itself is reflective, suppression removes the very primitives needed for $c _ { 1 }$ localization. Incorrect $c _ { 1 }$ outputs provide a direct diagnostic for this failure mode: when c1 centroid error exceeds 5 cm, task success drops below 30%.

## V. CONCLUSION

GST-VLA introduces two contributions for 3D-grounded VLA models. The Gaussian Spatial Tokenizer converts frozen depth and visual features into anisotropic 3D Gaussian tokens whose covariance eigenstructure encodes surface orientation and whose learned opacity encodes geometric confidence, with spatial attention pooling concentrating representational capacity on task-relevant geometry. Depth-Aware Chain-of-Thought reasoning supervises explicit 3D spatial verbalization as an intermediate generation target, with cross-attention to the raw Gaussian field providing geometric access at full resolution during each thought. The composite training loss couples the CoT and depth objectives with the Gaussian field parameters, creating synergistic gradient pathways between reasoning quality and geometric calibration. Evaluations and ablations confirm that each component contributes independently and that their combination provides super additive gains concentrated on precision demanding manipulation tasks.

## REFERENCES

[1] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al., âOpenvla: An opensource vision-language-action model,â in Conference on Robot Learning (CoRL). PMLR, 2024, pp. 2679â2713.

[2] Q. Zhao, Y. Lu, M. J. Kim, Z. Fu, Z. Zhang, Y. Wu, Z. Li, Q. Ma, S. Han, C. Finn, et al., âCot-vla: Visual chain-of-thought reasoning for vision-language-action models,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 1702â1713.

[3] D. Qu, H. Song, Q. Chen, Y. Yao, X. Ye, Y. Ding, Z. Wang, J. Gu, B. Zhao, D. Wang, et al., âSpatialvla: Exploring spatial representations for visual-language-action model,â arXiv preprint arXiv:2501.15830, 2025.

[4] T. Yuan, Y. Liu, C. Lu, Z. Chen, T. Jiang, and H. Zhao, âDepthvla: Enhancing vision-language-action models with depth-aware spatial reasoning,â arXiv preprint arXiv:2510.13375, 2025.

[5] H. Zhen, X. Qiu, P. Chen, J. Yang, X. Yan, Y. Du, Y. Hong, and C. Gan, â3d-vla: A 3d vision-language-action generative world model,â in International Conference on Machine Learning (ICML). PMLR, 2024, pp. 61 229â61 245.

[6] B. Kerbl, G. Kopanas, T. LeimkÃ¼hler, and G. Drettakis, â3d gaussian splatting for real-time radiance field rendering,â ACM Transactions on Graphics, vol. 42, no. 4, 2023.

[7] H. Jiang et al., âGausstr: Foundation model-aligned gaussian transformer for self-supervised 3d spatial understanding,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[8] Z. Zhang, R. Wu, L. Sun, and L. Zhang, âGpstoken: Gaussian parameterized spatially-adaptive tokenization for image representation and generation,â 2025. [Online]. Available: https://arxiv.org/abs/2509. 01109

[9] M. Zawalski, W. Chen, K. Pertsch, O. Mees, C. Finn, and S. Levine, âRobotic control via embodied chain-of-thought reasoning,â arXiv preprint arXiv:2407.08693, 2024.

[10] Q. Li, Y. Liang, Z. Wang, L. Luo, X. Chen, M. Liao, F. Wei, Y. Deng, S. Xu, Y. Zhang, et al., âCogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation,â arXiv preprint arXiv:2411.19650, 2024.

[11] J. Liu, H. Chen, P. An, Z. Liu, R. Zhang, C. Gu, X. Li, Z. Guo, S. Chen, M. Liu, et al., âHybridvla: Collaborative diffusion and autoregression in a unified vision-language-action model,â arXiv preprint arXiv:2503.10631, 2025.

[12] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. NieÃner, âScannet: Richly-annotated 3d reconstructions of indoor scenes,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 5828â5839.

[13] M. Roberts, J. Ramapuram, A. Ranjan, A. Kumar, M. A. Bautista, N. Paczan, R. Webb, and J. M. Susskind, âHypersim: A photorealistic synthetic dataset for holistic indoor scene understanding,â in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 10 912â10 922.

[14] G. Baruch, Z. Chen, A. Dehghan, T. Dimry, Y. Feigin, P. Fu, T. Gebauer, B. Joffe, D. Kurz, A. Schwartz, et al., âArkitscenes: A diverse realworld dataset for 3d indoor scene understanding using mobile rgb-d data,â arXiv preprint arXiv:2111.08897, 2021.

[15] X. Li, K. Hsu, J. Gu, K. Pertsch, O. Mees, H. R. Walke, C. Fu, I. Lunawat, I. Sieh, S. Kirmani, S. Levine, J. Wu, C. Finn, H. Su, Q. Vuong, and T. Xiao, âEvaluating real-world robot manipulation policies in simulation,â arXiv preprint arXiv:2405.05941, 2024.

[16] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone, âLibero: Benchmarking knowledge transfer for lifelong robot learning,â Advances in Neural Information Processing Systems (NeurIPS), 2023.

[17] X. Zhou, Y. Xu, G. Tie, Y. Chen, G. Zhang, D. Chu, P. Zhou, and L. Sun, âLibero-pro: Towards robust and fair evaluation of vision-languageaction models beyond memorization,â arXiv preprint arXiv:2510.03827, 2025.

## APPENDIX

The results presented in this paper are preliminary. Please note that the experiments are currently ongoing, and the final data is subject to change upon the completion of the study. All ideas, results, methods, and any content herein are the sole property of the authors. Reuse, reproduction, distribution, or any other use without explicit written permission from the authors is strictly prohibited. All rights reserved.

## ACKNOWLEDGMENT

This work was supported by Regional Innovation System & Education(RISE) program[B0080529002330], through the Gyeongbuk RISE CENTER, funded by the Ministry of Education(MOE) and the Gyeongsangbuk-do, Republic of Korea.(2025-RISE-15-115)