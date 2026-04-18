# Fading the Digital Ink: A Universal Black-Box Attack Framework for 3DGS Watermarking Systems

Qingyuan Zeng1, Shu Jiang1, Jiajing Lin2, Zhenzhong Wang2, Kay Chen Tan3, Min Jiang2\*

1 Institute of Artificial Intelligence, Xiamen University, China

2 School of Informatics, Xiamen University, China

3 Department of Data Science and Artificial Intelligence, The Hong Kong Polytechnic University, Hong Kong SAR

## Abstract

With the rise of 3D Gaussian Splatting (3DGS), a variety of digital watermarking techniques, embedding either 1D bitstreams or 2D images, are used for copyright protection. However, the robustness of these watermarking techniques against potential attacks remains underexplored. This paper introduces the first universal black-box attack framework, the Group-based Multi-objective Evolutionary Attack (GMEA), designed to challenge these watermarking systems. We formulate the attack as a large-scale multi-objective optimization problem, balancing watermark removal with visual quality. In a black-box setting, we introduce an indirect objective function that blinds the watermark detector by minimizing the standard deviation of features extracted by a convolutional network, thus rendering the feature maps uninformative. To manage the vast search space of 3DGS models, we employ a group-based optimization strategy to partition the model into multiple, independent sub-optimization problems. Experiments demonstrate that our framework effectively removes both 1D and 2D watermarks from mainstream 3DGS watermarking methods while maintaining high visual fidelity. This work reveals critical vulnerabilities in existing 3DGS copyright protection schemes and calls for the development of more robust watermarking systems.

## Introduction

3D Gaussian Splatting (3DGS) is an emerging 3D scene representation and reconstruction technology. It has the advantages of high fidelity, fast rendering speed, and real-time rendering capabilities (Kerbl et al. 2023; Wu et al. 2024b). It has broad application prospects in fields such as film production, game development, virtual reality, and autonomous driving (Zhu et al. 2025; Tu et al. 2025; Chen et al. 2025b). Given that creating a 3DGS model represents a significant investment in data acquisition, engineering expertise, and computational resources (Zhang et al. 2024a), and the copyright of 3DGS assets is prone to unauthorized distribution and malicious tampering, it is crucial to effectively protect the copyright of 3DGS assets. To meet this challenge, various invisible watermarking methods for 3DGS have been proposed (Chen et al. 2025a; Jang et al. 2025; Huang et al. 2025). They embed invisible copyright information directly into the Gaussian parameters of 3DGS models, which is subsequently extracted from the rendered images. These approaches encompass different strategies, such as encoding one-dimensional (1D) copyright strings (Chen et al. 2025a) or hiding entire two-dimensional (2D) data like logos or images as watermarks (Zhang et al. 2024b).

These 3DGS invisible watermarking methods (Chen et al. 2025a; Huang et al. 2025) need to be robust enough to truly protect the copyright of 3D assets in reality. That is, they need to maintain the integrity and detectability of watermarks in the face of various potential attacks (Cox et al. 1997; Zhao et al. 2024). However, so far, no research has explored the robustness of 3DGS invisible watermarks when facing attacks. Therefore, this paper aims to answer the following question: is there a universal attack method that can destroy these 3DGS invisible watermarks?

There are the following difficulties in destroying 3DGS invisible watermarks. First, visual fidelity must be preserved. The attacker is required to remove the watermark without noticeably degrading the quality of the 3DGS model (Zhang et al. 2020). Second, attacks typically occur in a black-box setting, where attackers often lack knowledge of the watermarkâs content, embedding process, and detection process (Papernot et al. 2017; Zeng et al. 2024). This means an attacker cannot accurately locate the watermarkâs distribution in the Gaussian parameters or rendered images, nor can they use the watermark detectorâs gradients to guide the optimization process through backpropagation.

To narrow this research gap, we propose GMEA, a universal black-box attack framework against 3DGS invisible watermarks. Our approach formulates the attack as a largescale multi-objective optimization problem (Wang et al. 2024), seeking to simultaneously destroy the watermark while preserving the modelâs visual quality (Wang et al. 2021). The decision variables represent the attackerâs two primary actions: selectively pruning some of the modelâs Gaussian kernels and subtly shifting the color values of others. The optimization objectives are: 1) minimizing visual quality degradation measured by MSE between original and perturbed renders, and 2) maximizing watermark destruction. Critically, in the black-box setting where watermark detectors are inaccessible, we introduce an indirect objective function to evaluate watermark destruction: the standard deviation of convolutional feature maps extracted from rendered images (Lu et al. 2020). Minimizing this deviation reduces discriminative information in feature maps, effectively blinding downstream watermark decoders that rely on convolutional features to extract watermark signals. This breaks the watermark extraction process despite having no detector access.

<!-- image-->  
Figure 1: Illustration of the group-based optimization strategy. The original 3DGS model (full Lego object) is partitioned into multiple sub-optimization problems (individual Lego components) to manage the large search space effectively.

To solve this large-scale multi-objective problem, we build GMEA upon an evolutionary algorithm, inspired by its unique ability to handle such complex optimizations (Deb et al. 2002; Hong, Jiang, and Yen 2023; Liang et al. 2023; Wang et al. 2024). However, a direct application of evolutionary algorithms is computationally inefficient. Due to the oversized search space created by the vast number of Gaussian kernels, the process of converging to an effective solution is slow. Therefore, to make the optimization tractable and efficient, we break down the large-scale optimization problem into several smaller sub-optimization problems to be solved independently (Zhang et al. 2018), as shown in Figure 1. Then, we combine the solutions of each suboptimization problem to obtain the solution to the original optimization problem. Specifically, as shown in Figure 2, we use unsupervised clustering algorithms (such as K-Means (MacQueen 1967; Van Gansbeke et al. 2020)) to cluster the Gaussian kernels of the watermarked 3DGS model into k clusters from the perspective of position, obtaining k sub-3DGS models. Then, we perform multi-objective evolutionary algorithm on each sub-3DGS model to remove the watermark. Finally, we merge all the optimized sub-3DGS models to obtain the complete 3DGS model without watermarks. Our contributions can be summarized as follows:

1. We propose the first universal black-box attack framework GMEA against 3DGS invisible watermarks. It features a model-agnostic objective that disables watermark detection by disrupting convolutional features without requiring any knowledge of the detector.

2. We design a group-based optimization strategy that partitions the immense search space of 3DGS models, significantly improving the search efficiency of our evolutionary algorithm in discovering effective solutions.

3. We conduct extensive experiments to validate our frameworkâs effectiveness and universality, successfully attacking leading methods for both 1D and 2D 3DGS watermarking.

## Related Works

## 3D Gaussian Splatting

3D Gaussian Splatting (3DGS) is a foundational technique for real-time, photo-realistic rendering, with its applications rapidly expanding (Kerbl et al. 2023; Zhu et al. 2024; Bao et al. 2025). In 3D content generation, frameworks like GaussianDreamer utilize 2D diffusion priors for automated asset creation from text or images (Yi et al. 2024). For dynamic scenes, 4D Gaussian Splatting models complex, non-rigid motions via time-varying deformation fields (Wu et al. 2024a). Its utility extends to geometric reconstruction, where methods like SuGaR extract detailed meshes by imposing local geometric constraints (Guedon and Lepetit Â´ 2024). Furthermore, PhysGaussian enables realistic physical interactions by treating Gaussians as Lagrangian particles within a physics simulator (Xie et al. 2024).

As 3DGS integrates with physics engines and generative models, becoming more complex and less transparent, thereâs an urgent need in academia and industry to establish copyright protection, robustness analysis, and trustvalidation systems for these models.

## 3D Gaussian Splatting Watermarking

Modern 3DGS watermarking techniques imperceptibly embed copyright data while preserving visual fidelity. Approaches are typically categorized by message dimensionality: 1D bitstreams for simple copyright strings, and 2D messages for complex data like logos or images.

1D Bitstream Watermarking. These methods focus on embedding one-dimensional (1D) binary messages, such as copyright strings, directly into the 3DGS model. GuardSplat (Chen et al. 2025a) embeds messages by modifying existing Gaussian kernels. It utilizes small, learnable offsets to the Spherical Harmonic (SH) features to integrate the watermark, a technique designed to preserve the modelâs original 3D structure and visual fidelity. 3D-GSW (Jang et al. 2025) takes a refinement approach, preparing the model for watermarking through a process called Frequency-Guided Densification (FGD). This technique first prunes Gaussians that have minimal impact on rendering quality and then splits Gaussians located in high-frequency areas to maintain visual fidelity. The watermark is subsequently embedded into this optimized set of Gaussians via fine-tuning. Gaussian-Marker (Huang et al. 2025) employs an additive strategy, leaving original Gaussians untouched. It identifies regions of high uncertainty in the model and introduces new, dedicated Gaussian kernels called âGaussianMarkersâ within these areas to carry the watermark information.

<!-- image-->  
Figure 2: An overview of our proposed Group-based Multi-objective Evolutionary Attack (GMEA) framework. The attack pipeline begins by partitioning the watermarked 3DGS model into several spatially coherent sub-3DGS. Each sub-3DGS then undergoes an independent multi-objective evolutionary optimization. During this phase, potential modificationsârepresented by a binary mask for pruning Gaussians and a perturbation vector for altering colorsâare evolved to simultaneously minimize visual quality loss $( \bar { F _ { 1 } } )$ and maximize watermark destruction $( F _ { 2 } )$ . The resulting optimized and unwatermarked sub-3DGS are then reassembled to form the final unwatermarked 3DGS model.

2D Message Watermarking. Another approach pushes the capacity of steganography further by enabling the embedding of two-dimensional (2D) messages, such as entire images or logos. GS-Hider (Zhang et al. 2024b) exemplifies this category. It achieves high-capacity message hiding by fundamentally altering the rendering pipeline. Instead of just modifying SH coefficients, it replaces them entirely with a coupled secured feature attribute. This high-dimensional feature is then rendered into a feature map. Two parallel decoders are employed: a public scene decoder that reconstructs the original visual scene, and a private message decoder that extracts the hidden 2D image from the same feature map. This decoupling allows for the concealment of complex messages without direct interference with the primary sceneâs rendering.

While the practicality of watermarking methods depends on robustness, the resilience of 3DGS schemes against dedicated attacks remains largely untested. We therefore introduce the first universal attack framework to serve as a security benchmark for the 3DGS ecosystem. Our work aims to not only assess current vulnerabilities but also to catalyze the development of more secure future solutions.

## Methodology

In this section, we detail our proposed black-box attack framework, Group-based Multi-objective Evolutionary Attack (GMEA), designed to remove invisible watermarks from 3DGS models. The overall pipeline of our method is illustrated in Figure 2, while the complete pseudocode and theoretical proofs are provided in the Appendix.

## Problem Formulation

We aim to find an adversarial 3DGS model, $\mathbf { G } _ { a d v } ,$ derived from a watermarked 3DGS model $\mathbf { G } _ { w m } .$ The goal is to simultaneously maintain high visual fidelity and destroy the embedded watermark. This is formulated as a multiobjective optimization problem:

$$
\begin{array} { r l } { \underset { { \mathbf { G } } _ { a d v } } { \operatorname* { m i n } } } & { { } F _ { 1 } ( { \mathbf { G } } _ { a d v } , { \mathbf { G } } _ { w m } ) , F _ { 2 } ( { \mathbf { G } } _ { a d v } ) } \\ { \mathrm { s . t . } } & { { } { \mathbf { G } } _ { a d v } \in \mathcal { P } ( { \mathbf { G } } _ { w m } ) } \end{array}\tag{1}
$$

Here, $F _ { 1 }$ measures visual quality loss and $F _ { 2 }$ quantifies watermark destruction. The constraint $\mathbf { G } _ { a d v } \ \in \ \bar { \mathcal { P } } ( \mathbf { G } _ { w m } )$ specifies that any candidate solution ${ \bf G } _ { a d v }$ must be generated by applying the allowed perturbations to $\mathbf { G } _ { w m }$

## Group-Based Optimization Strategy

Our group-based optimization strategy tackles the immense search space of 3DGS models by decomposing the task into smaller, spatially coherent sub-problems. This partitioning makes the search for an effective solution more efficient by simplifying the optimization landscape.

To achieve this partitioning, we employ the K-Means clustering on the spatial coordinates (xyz) of the Gaussian kernels. Let the set of all 3D coordinates for the N Gaussians in the watermarked model $\mathbf { G } _ { w m }$ be $\mathbf { P } = \left\{ \mathbf { p } _ { 1 } , \mathbf { p } _ { 2 } , \ldots , \mathbf { p } _ { N } \right\}$ , where $\mathbf { p } _ { j } \in \mathbb { R } ^ { 3 }$ . The goal of K-Means is to partition this set of kernels P into k disjoint spatial clusters, denoted by $\textbf { S } = ~ \{ S _ { 1 } , S _ { 2 } , \ldots , S _ { k } \}$ , by minimizing the within-cluster sum of squares. The minimization criterion is formalized as:

$$
a r g m i n \sum _ { S } ^ { k } \sum _ { j \in S _ { i } } \| \mathbf { p } _ { j } - \pmb { \mu } _ { i } \| ^ { 2 } ,\tag{2}
$$

where $\mu _ { i }$ is the geometric centroid of the coordinates in cluster $S _ { i }$

Once the optimal coordinate clusters $\{ S _ { 1 } , \ldots , S _ { k } \}$ are determined, we map these spatial groupings back to the Gaussiansâ original indices. This creates a corresponding partition of the index set $\{ 1 , 2 , \ldots , N \}$ into k disjoint sets $\mathbf { \bar { \{ } }  I _ { 1 } , \ldots , I _ { k } \}$ . Each index set $I _ { i }$ is formalized as

$$
I _ { i } = \{ j \mid \mathbf { p } _ { j } \in S _ { i } \} .\tag{3}
$$

With the indices properly partitioned, we formally construct the 3DGS sub-models. The i-th sub-model, $\mathbf { G } _ { w m } ^ { ( i ) }$ , is defined as the collection of Gaussian kernels whose indices fall into the set $I _ { i } { \mathrm { : } }$ :

$$
\mathbf { G } _ { w m } ^ { ( i ) } = \{ g _ { j } \in \mathbf { G } _ { w m } \mid j \in I _ { i } \} ,\tag{4}
$$

where $g _ { j }$ represents the j-th Gaussian kernel. This decomposition of the large-scale task into smaller sub-problems significantly reduces search complexity, enabling a more efficient optimization.

## Multi-objective Evolutionary Attack

Each sub-problem defined by a sub-model $\mathbf { G } _ { w m } ^ { ( i ) }$ is solved with a multi-objective evolutionary algorithm to find an effective adversarial solution , $\mathbf { G } _ { a d v } ^ { ( i ) }$ . The algorithm refines a population of solutions to approximate the Pareto-optimal front, balancing the conflicting objectives of visual fidelity and watermark removal.

Individual Representation. Each individual in our population represents a potential modification to a sub-model $\mathbf { G } _ { w m } ^ { ( i ) }$ containing $N ^ { ( i ) }$ Gaussians. The individualâs genetic representation is a vector

$$
\mathbf { x } ^ { ( i ) } = [ \mathbf { m } ^ { ( i ) } , \mathbf { c } ^ { ( i ) } ] ,\tag{5}
$$

which concatenates a binary mask vector m ${ \mathbf \Lambda } ^ { ( i ) } \in \{ 0 , 1 \} ^ { N ^ { ( i ) } }$ and a color perturbation vector $\mathbf { c } ^ { ( i ) } \ \in \ [ - \epsilon , \epsilon ] ^ { 3 N ^ { ( i ) } }$ . The mask vector determines which Gaussian kernels are pruned $( m _ { j } = 0 )$ , while the color perturbation vector defines additive shifts to the DC color component (RGB) for the remaining Gaussian kernels.

The adversarial sub-model $\mathbf { G } _ { a d v } ^ { ( i ) }$ is constructed by applying these modifications. Let $\mathbf { C } _ { d c }$ be the $N ^ { ( i ) } \times 3$ matrix of original DC colors. The new color matrix, $\mathbf { C } _ { d c , a d v } ,$ is computed as:

$$
\begin{array} { r } { \mathbf { C } _ { d c , a d v } = \operatorname { d i a g } ( \mathbf { m } ^ { ( i ) } ) \left( \mathbf { C } _ { d c } + \mathrm { R e s h a p e } ( \mathbf { c } ^ { ( i ) } ) \right) . } \end{array}\tag{6}
$$

Here, the diagonalized mask diag $\mathbf { \Lambda } _ { \mathbf { \Lambda } ^ { \prime } } ^ { \prime } ( \mathbf { m } ^ { ( i ) } )$ filters the Gaussian kernels, while the reshaped perturbation vector modifies the colors of the survivors. All other Gaussian parameters (e.g., position, scale, opacity) are similarly filtered by the mask. The resulting adversarial sub-model $\mathbf { G } _ { a d \iota } ^ { ( i ) }$ is then used to render images $\mathbf { R } _ { a d v } ^ { ( i ) }$ for fitness evaluation.

Objective Functions. We evaluate each individualâs fitness using two conflicting objectives designed to preserve visual fidelity while removing the watermark.

The first objective, visual quality loss $( F _ { 1 } )$ , measures the perceptual difference between the original watermarked 3DGS and adversarial 3DGS. We calculate this loss as a weighted combination of the L1 distance and the Structural Similarity Index Measure (SSIM) over images rendered from multiple viewpoints $\{ v _ { 1 } , \ldots , v _ { N _ { v } } \}$

$$
\begin{array} { r } { F _ { 1 } ( \mathbf { G } _ { a d v } ^ { ( i ) } ) = \cfrac { 1 } { N _ { v } } \displaystyle \sum _ { v = 1 } ^ { N _ { v } } \Big [ \lambda \mathcal { L } _ { \mathrm { L 1 } } ( \mathbf { R } _ { a d v } ^ { ( i , v ) } , \mathbf { R } _ { w m } ^ { ( i , v ) } ) } \\ { + ( 1 - \lambda ) ( 1 - \mathrm { S S I M } ( \mathbf { R } _ { a d v } ^ { ( i , v ) } , \mathbf { R } _ { w m } ^ { ( i , v ) } ) ) \Big ] , } \end{array}\tag{7}
$$

where $\mathbf { R } _ { a d v } ^ { ( i , v ) }$ and $\mathbf { R } _ { w m } ^ { ( i , v ) }$ are the rendered images of adversarial 3DGS and original watermarked 3DGS. Î» is a weighting factor. Minimizing $F _ { 1 }$ guides solutions to be visually indistinguishable from the original.

The second objective, watermark destruction $( F _ { 2 } ) .$ , provides a model-agnostic attack by neutralizing the convolutional feature extraction common to all decoders. We achieve this by minimizing the feature mapsâ statistical variance, which flattens their patterns and renders them non-discriminative for watermark detection. Specifically, we pass a rendered adversarial image $\mathbf R _ { a d v } ^ { ( i , v ) }$ through a convolutional feature extractor Î¦. The dispersion of a single feature channel, $D ( \mathbf { F } _ { c } )$ , is then quantified by its standard deviation:

$$
D ( \mathbf { F } _ { c } ) = \sqrt { \frac { 1 } { H ^ { \prime } W ^ { \prime } } \sum _ { h , w } ( \mathbf { F } _ { c } ( h , w ) - \bar { \mathbf { F } } _ { c } ) ^ { 2 } } ,\tag{8}
$$

where $\bar { \mathbf { F } } _ { c }$ is the mean activation of channel c. The final objective $F _ { 2 }$ is the average dispersion over all feature channels and viewpoints:

$$
F _ { 2 } ( \mathbf { G } _ { a d v } ^ { ( i ) } ) = \frac { 1 } { N _ { v } } \sum _ { v = 1 } ^ { N _ { v } } \left[ \frac { 1 } { C ^ { \prime } } \sum _ { c = 1 } ^ { C ^ { \prime } } D ( \Phi ( \mathbf { R } _ { a d v } ^ { ( i , v ) } ) _ { c } ) \right] .\tag{9}
$$

Evolutionary Process. The evolutionary process begins with a population $\mathcal { P } _ { t }$ of size $N _ { p o p }$ . At each generation t, an offspring population $\mathcal { Q } _ { t }$ is generated from the current population $\mathcal { P } _ { t }$ . This involves two main operators. First, a crossover operator produces two new solutions from a pair of parents $( { \bf x } _ { a } , { \bf x } _ { b } )$ , distributing the offspring around the parentsâ positions in the search space:

$$
\mathbf { x } _ { a , b } ^ { \prime } = 0 . 5 \left[ ( \mathbf { x } _ { a } + \mathbf { x } _ { b } ) \mp \beta | \mathbf { x } _ { b } - \mathbf { x } _ { a } | \right] ,\tag{10}
$$

where $\beta$ is a hyperparameter controlling the spread of the offspring. Subsequently, a mutation operator introduces finegrained perturbations to an individual $\mathbf { x } ^ { \prime }$ to enhance local exploration:

$$
x _ { j } ^ { \prime \prime } = x _ { j } ^ { \prime } + \eta _ { j } ( u b _ { j } - l b _ { j } ) ,\tag{11}
$$

<!-- image-->  
Figure 3: Qualitative results of the GMEA attack on the Drums (a-f) and Flower (g-l) scenes. While a single-objective attack GMEA (w/o F1) corrupts the watermark at the cost of severe visual degradation (c,d,i,j), our full multi-objective approach GMEA successfully removes the watermark while preserving high visual fidelity (e,f,k,l).

where $\eta _ { j }$ is a small perturbation value, and $[ l b _ { j } , u b _ { j } ]$ represents the defined lower and upper bounds for the j-th decision variable.

After generating offspring, a rigorous selection process determines which individuals will form the next generationâs population. First, the current parent population $( \mathcal { P } _ { t } )$ and their offspring (Qt) are merged into a combined pool, $\mathcal { R } _ { t } = \mathcal { P } _ { t } \cup \mathcal { Q } _ { t }$ . This pool is then ranked and partitioned into a hierarchy of non-dominated fronts $\{ \mathcal { F } _ { 1 } , \mathcal { F } _ { 2 } , \ldots \}$ based on Pareto dominance (Hong, Jiang, and Yen 2023).

The next generation is formed by elitism, admitting individuals from the best non-dominated fronts $( \mathcal { F } _ { 1 } , \mathcal { F } _ { 2 } , \ldots )$ until the population capacity $( N _ { p o p } )$ is met. To maintain diversity when the final front $( \mathcal { F } _ { l } )$ is truncated, we rank its members by a density metric that prioritizes solutions in less crowded regions. The density score for an individual solution x, denoted $d ( \mathbf { x } )$ , is calculated as:

$$
d ( \mathbf { x } ) = \sum _ { o = 1 } ^ { M } \frac { F _ { o } ( \mathrm { n e i g h b o r } ^ { + } ) - F _ { o } ( \mathrm { n e i g h b o r } ^ { - } ) } { F _ { o } ^ { \operatorname* { m a x } } - F _ { o } ^ { \operatorname* { m i n } } } ,\tag{12}
$$

where M is the number of objectives, and $F _ { o } ( \mathrm { n e i g h b o r ^ { \pm } } )$ are the objective values of the neighbors of solution x after sorting the front along objective o. Individuals with higher density scores are chosen to fill the remaining slots, forming a diverse parent population for the next evolutionary cycle.

Reconstructing the Adversarial Model. The final step is to reconstruct the complete adversarial model, ${ \bf G } _ { a d v }$ . Since our group-based strategy operates on disjoint sets of Gaussian kernels, this reconstruction is a straightforward union of the k optimized sub-models:

$$
\mathbf { G } _ { a d v } = \bigcup _ { i = 1 } ^ { k } \mathbf { G } _ { a d v } ^ { ( i ) } .\tag{13}
$$

The resulting model $\mathbf { G } _ { a d v }$ aggregates all modifications from the independent optimization runs and represents the final output of our attack framework.

## Evaluation and Results

## Experimental Settings

Model and dataset. To demonstrate GMEAâs versatility, we target representative systems from two distinct categories of 3DGS watermarking: GaussianMarker (Huang et al. 2025) for 1D bitstream watermarks and GS-Hider (Zhang et al. 2024b) for 2D image watermarks. The evaluation was performed on two datasets: Blender dataset (Mildenhall et al. 2021), comprising objects without backgrounds, and the more challenging LLFF dataset (Mildenhall et al. 2019), which features complex real-world scenes.

Evaluation metrics. We assess our frameworkâs performance based on two criteria: visual fidelity and watermark removal efficacy. Visual quality is quantified using standard image metrics: Peak Signal-to-Noise Ratio (PSNR), the Structural Similarity Index Measure (SSIM), and Mean Squared Error (MSE) (Setiadi 2021). For evaluating 1D watermark removal, we use the standard Bit Accuracy Rate (BAR) and our proposed Watermark Uncertainty Score (WUS) and Information Destruction Score (IDS). Detailed definitions for these metrics, along with further experimental settings and results, are deferred to the Appendix.

## Experiment Results

Attack Performance Evaluation. We assessed our GMEA frameworkâs effectiveness through extensive experiments on 1D and 2D watermarking systems, as shown in Table 1. Our full GMEA attack significantly disrupts the nearperfect watermark extraction of the No Attack baseline in the 1D watermarking system, reducing the Bit Accuracy Rate (BAR) to an average of approximately 65% and substantially increasing the Watermark Uncertainty Score (WUS) and Information Destruction Score (IDS). This renders the extracted bitstream highly unreliable.

In the 2D watermarking system, our full GMEA methodâs success is evident in the degradation of the extracted watermark imageâs quality, with a drastic drop in both SSIM and

<!-- image-->

<!-- image-->

<!-- image-->  
00101111,01010100, 11010011,01011100, 01100010,10100100

<!-- image-->  
Figure 4: The GMEA attackâs Pareto front, showing the trade-off between visual fidelity and attack success. The main plot highlights the optimizationâs improvement. Three solutions are visualizedâ(a) quality-optimal, (c) attack-optimal, and (b) balancedâshowing their resulting render and the corrupted watermark with errors marked in red.

Table 1: This table quantitatively evaluates the attackâs performance on 1D and 2D watermarking systems. Each row pairs a Blender scene with an LLFF scene, presenting their respective results in corresponding columns.
<table><tr><td></td><td></td><td colspan="6">1D Watermark (GaussianMarker)</td><td colspan="6">2D Watermark (GS-Hider)</td></tr><tr><td></td><td></td><td colspan="3">Blender</td><td colspan="3">LLFF</td><td colspan="3">Blender</td><td colspan="3">LLFF</td></tr><tr><td>Scene</td><td>Method</td><td>BARâ</td><td>WUSâ</td><td>IDSâ</td><td>BARâ</td><td>WUSâ</td><td>IDSâ</td><td>SSIMâ</td><td>PSNRâ</td><td>MSEâ</td><td>SSIMâ</td><td>PSNRâ</td><td>MSEâ</td></tr><tr><td rowspan="3">Chair / Fern</td><td>No Attack</td><td>99.95</td><td>0.1</td><td>0.1</td><td>100</td><td>0.0</td><td>0.0</td><td>91.18</td><td>15.74</td><td>0.026</td><td>97.86</td><td>26.43</td><td>0.002</td></tr><tr><td>GMEA (w/o F1)</td><td>45.99</td><td>89.1</td><td>89.03</td><td>50.33</td><td>98.21</td><td>97.78</td><td>52.31</td><td>5.16</td><td>0.305</td><td>667.17</td><td>9.37</td><td>0.115</td></tr><tr><td>GMEA (ours)</td><td>67.44</td><td>65.12</td><td>64.87</td><td>56.94</td><td>86.11</td><td>85.98</td><td>74.63</td><td>10.26</td><td>0.094</td><td>70.61</td><td>9.51</td><td>0.1111</td></tr><tr><td rowspan="3">Drums / Flower</td><td>No Attack</td><td>99.71</td><td>0.58</td><td>0.57</td><td>100</td><td>0.0</td><td>0.0</td><td>93.12</td><td>18.12</td><td>0.015</td><td>96.53</td><td>23.67</td><td>0.004</td></tr><tr><td>GMEA (w/o F1)</td><td>45.58</td><td>889.96</td><td>90.07</td><td>57.92</td><td>84.17</td><td>83.9</td><td>52.81</td><td>5.03</td><td>00.314</td><td>62.48</td><td>8.49</td><td>00.141</td></tr><tr><td>GMEA (ours)</td><td>65.0</td><td>70.0</td><td>69.87</td><td>60.83</td><td>78.33</td><td>78.25</td><td>75.4</td><td>9.92</td><td>0.102</td><td>65.42</td><td>11.86</td><td>0.065</td></tr><tr><td rowspan="3">Ficus / Fortress</td><td>No Attack</td><td>96.06</td><td>7.87</td><td>7.87</td><td>100</td><td>0.0</td><td>0.0</td><td>95.17</td><td>22.71</td><td>0.005</td><td>94.18</td><td>19.2</td><td>0.012</td></tr><tr><td>GMEA (w/o F1)</td><td>54.32</td><td>89.</td><td>888.67</td><td>57.45</td><td>80.15</td><td>80..3</td><td>64.88</td><td>7.2</td><td>0.19</td><td>30.32</td><td>5.93</td><td>0.254</td></tr><tr><td>GMEA (ours)</td><td>71.98</td><td>56.04</td><td>55.84</td><td>64..58</td><td>70.3</td><td>71.1</td><td>74.88</td><td>8.63</td><td>0.137</td><td>56.24</td><td>8.99</td><td>0.126</td></tr><tr><td rowspan="3">Hotdog / Horns</td><td>No Attack</td><td>98.61</td><td>2.77</td><td>2.73</td><td>100</td><td>0.0</td><td>0.0</td><td>95.65</td><td>22.71</td><td>0.005</td><td>97.86</td><td>26.43</td><td>0.002</td></tr><tr><td>GMEA (w/o F1)</td><td>58.16</td><td>82.77</td><td>82.85</td><td>60.36</td><td>77.81</td><td>77.63</td><td>60.36</td><td>6.63</td><td>0.217</td><td>60.09</td><td>8.7</td><td>0.135</td></tr><tr><td>GMEA (ours)</td><td>58.21</td><td>83.17</td><td>83.11</td><td>61.98</td><td>76.04</td><td>75.9</td><td>73.7</td><td>9.66</td><td>0.108</td><td>75.75</td><td>12.68</td><td>0.053</td></tr><tr><td rowspan="3">Lego / Leaves</td><td>No Attack</td><td>99.99</td><td>0.02</td><td>0.02</td><td>100</td><td>0.0</td><td>0.0</td><td>94.33</td><td>21.19</td><td>0.007</td><td>98.54</td><td>29.93</td><td>0.001</td></tr><tr><td>GMEA (w/o F1)</td><td>45.19</td><td>88.21</td><td>88.03</td><td>660.13</td><td>78.21</td><td>78.15</td><td>68.98</td><td>6.76</td><td>0.211</td><td>71.73</td><td>11.39</td><td>0.072</td></tr><tr><td>GMEA (ours)</td><td>68.53</td><td>67.35</td><td>65.88</td><td>68.75</td><td>62.5</td><td>62.47</td><td>74.45</td><td>10.24</td><td>0.094</td><td>80.2</td><td>13.26</td><td>0.047</td></tr><tr><td rowspan="3">Materials / Trex</td><td></td><td>99.43</td><td>1.15</td><td>1.13</td><td>100</td><td>0.0</td><td>0.0</td><td>91.67</td><td>17.75</td><td>0.016</td><td>97.44</td><td>28.03</td><td>0.001</td></tr><tr><td>No Attack</td><td>55.05</td><td>83.69</td><td>83.75</td><td>59.45</td><td>77.25</td><td>77.41</td><td>60.22</td><td>6.25</td><td>0.237</td><td>64.96</td><td></td><td>0.121</td></tr><tr><td>GMEA (w/o F1) GMEA (ours)</td><td>63.98</td><td>71.96</td><td>71.93</td><td>65.63</td><td>68.75</td><td>68.4</td><td>79.07</td><td>10.33</td><td>0.092</td><td>69.88</td><td>9.15 10.95</td><td>0.081</td></tr><tr><td rowspan="3">Mic / Room</td><td></td><td>99.21</td><td>1.58</td><td>1.57</td><td>100</td><td>0.0</td><td>0.0</td><td></td><td></td><td>0.013</td><td>98.28</td><td>29.74</td><td>0.001</td></tr><tr><td>No Attack</td><td>56.22</td><td>86.44</td><td>86.46</td><td>66.14</td><td>65.43</td><td>65.58</td><td>92.7 32.52</td><td>18.64 2.19</td><td>0.603</td><td>70.4</td><td></td><td>0.109</td></tr><tr><td>GMEA (w/o F1) GM(ours)</td><td>67.19</td><td>65.58</td><td>65.51</td><td>71.53</td><td>56.94</td><td>56.62</td><td>77.79</td><td>8.95</td><td>0.127</td><td>75.07</td><td>9.59 11.26</td><td>0.074</td></tr><tr><td rowspan="3">Ship / Orchids</td><td></td><td>99.4</td><td>1.21</td><td>1.19</td><td>100</td><td>0.0</td><td>0.0</td><td>95.54</td><td>21.41</td><td>0.007</td><td>98.68</td><td>31.89</td><td>0.001</td></tr><tr><td>No Attack</td><td>556.65</td><td>84.25</td><td>84.32</td><td>53.18</td><td>93.84</td><td>94.01</td><td>663.47</td><td>6.56</td><td>0.2</td><td>61.18</td><td></td><td></td></tr><tr><td>GMEA (w/o F1) GMEA (ours)</td><td>64.9</td><td>70.12</td><td>70.04</td><td>55.95</td><td>88.1</td><td>88.19</td><td>72.56</td><td>8.67</td><td>0.135</td><td>67.72</td><td>8.06 9.15</td><td>0.156 00.121</td></tr></table>

PSNR values compared to the No Attack baseline. This severe degradation, detailed in Table 1, confirms our attack effectively renders the 2D visual watermark unrecognizable. These findings validate our GMEAâs capability to successfully compromise the detectability of different mainstream watermarking schemes.

Ablation Study. To further analyze the components of our framework, we conducted an ablation study by removing the visual quality objective $( F _ { 1 } )$ and creating a single-objective variant, GMEA (w/o $F _ { 1 } )$ , which solely optimizes for watermark destruction $( F _ { 2 } )$ . As shown in Table 1, this ablation variant exhibits even more potent attack capabilities.

For the 1D watermark, GMEA (w/o $F _ { 1 } )$ achieves a BAR closer to 50% (the theoretical value for random guessing) and higher WUS/IDS scores than our full two-objective method GMEA. For instance, in the Materials / Trex scene, the WUS score for GMEA (w/o $F _ { 1 } )$ attack reaches 83.69, compared to 71.96 for the full GMEA. For the 2D watermark, the GMEA (w/o $F _ { 1 } )$ attack results in a significantly lower SSIM/PSNR for the extracted watermark image, indicating more severe corruption. This demonstrates that by focusing exclusively on maximizing watermark destruction without the constraint of preserving visual fidelity, the attack can achieve a higher degree of watermark removal. However, it completely disregards maintaining the visual quality of the 3DGS model, resulting in the 3DGS being unusable after watermark removal. A detailed analysis of the resulting visual degradation is presented in the next section.

Justification for the Multi-objective Approach. While the single-objective attack GMEA (w/o $\bar { F } _ { 1 } \bar { ) }$ offers superior watermark destruction, it comes at the unacceptable cost of visual degradation to the 3DGS model. This trade-off is qualitatively illustrated in Figure 3. As the figure shows, although the watermark extracted by GMEA (w/o $F _ { 1 } )$ is more severely distorted, the rendered image is also riddled with visual artifacts, unlike the pristine render from our full GMEA.

Convergence Curves for Objective Functions F1 & F2  
<!-- image-->

<!-- image-->  
Figure 5: Visualization of GMEAâs optimization. The top plot shows the convergence of objectives $F _ { 1 }$ and $F _ { 2 } .$ The bottom panels show the feature map becoming uniform, which visually confirms the destruction of the watermark.

Table 2: Assessing the visual distortion of 3DGS introduced by our attack via an ablation study.
<table><tr><td>Watermarking Target</td><td>Attack Method</td><td>SSIMâ</td><td>PSNRâ</td><td>MSEâ</td></tr><tr><td>1D (GaussianMarker)</td><td>GMEA (w/o F1) G (ours)</td><td>67.19 95.13</td><td>20.31 30.66</td><td>0.0112 00011</td></tr><tr><td rowspan="2">2D (GS-Hider)</td><td>GMEA (w/o F1)</td><td>60.51</td><td>17.19</td><td>0.0263</td></tr><tr><td>GMEA (urs)</td><td>98.22</td><td>37.76</td><td>0.002</td></tr></table>

The quantitative data presented in Table 2 corroborates this visual evidence. Our full GMEA method maintains high SSIM (above 95%) and PSNR values, indicating minimal distortion. In contrast, the GMEA (w/o $F _ { 1 } )$ variant causes a drastic drop in these metrics. Since an attack that destroys an assetâs visual value defeats the purpose of the theft, this ablation study validates the necessity of our multi-objective formulation, which effectively balances potent watermark removal with the preservation of high visual fidelity.

Optimization Dynamics and Convergence. Figure 5 visualizes the operational dynamics of our GMEA. The top graph shows the steady convergence of both visual quality loss $( F _ { 1 } )$ and watermark destruction loss $( F _ { 2 } )$ , demonstrating the effectiveness of our multi-objective evolutionary algorithm. The bottom panels qualitatively validate our watermark destruction objective $( F _ { 2 } )$ by displaying visualizations of the intermediate convolutional feature map. At Generation 0, the feature map exhibits distinct spatial patterns that are essential for watermark decoders. As the optimization progresses, these patterns are suppressed, and by Generation 200, the feature map is largely homogeneous, erasing its discriminative features. Since any potential downstream decoder must rely on these upstream convolutional features, this flattening effectively blinds the entire watermark extraction pipeline. This visual collapse of feature information directly corresponds to the convergence of the $F _ { 2 }$ curve, proving that minimizing feature variance is a successful and universal indirect attack strategy in a black-box setting.

Table 3: Time-memory trade-off analysis for our groupbased strategy. The table shows the computational time and peak GPU memory required to reach a fixed target fitness $\dot { ( } F _ { 1 } \approx 1 . 9 , F _ { 2 } \approx \dot { 1 . 7 } )$ as the number of groups (k) varies.
<table><tr><td>Groups (k)</td><td>Time (min)</td><td>GPU Memory (GB)</td></tr><tr><td></td><td>74.2</td><td>24.0</td></tr><tr><td>5</td><td>32.5</td><td>33.6</td></tr><tr><td>10</td><td>23.7</td><td>43.2</td></tr><tr><td>20</td><td>20.1</td><td>55.4</td></tr><tr><td>50</td><td>18.5</td><td>84.9</td></tr></table>

Pareto Front Analysis and Solution Trade-offs. A key strength of our multi-objective approach is its ability to find a set of optimal solutions representing the trade-offs between conflicting objectives. Figure 4 analyzes the trade-off between attack effectiveness (low $F _ { 2 } )$ and visual fidelity (low $F _ { 1 } )$ . The main plot shows a significant improvement, with solutions evolving from a suboptimal initial state (top-right, gray) to a superior final Pareto front (bottom-left, blue).

For a more granular analysis of the solution space, we visualize three representative solutions from the final front. The Attack-Optimal Solution (c) achieves the most effective watermark corruption at the cost of slight visual artifacts. Conversely, the Quality-Optimal Solution (a) yields the highest visual fidelity but with a less damaged watermark. The Balanced Trade-off Solution (b) presents an ideal compromise, achieving significant watermark disruption with negligible visual distortion. This analysis highlights GMEAâs superiority: it discovers a range of effective attack solutions and offers the flexibility to select one based on the desired balance between efficacy and fidelity.

Analysis of the Group-Based Strategy. We analyzed the impact of the number of groups (k) on optimization efficiency, with results shown in Table 3. The data reveals a clear time-memory trade-off: increasing k accelerates convergence by parallelizing the search, but at the cost of higher peak GPU memory. Considering this trade-off, we identify k = 10 as a balanced setting for our main experiments, as it provides a significant speedup while maintaining a manageable memory footprint. More analysis is in the Appendix.

## Conclusion

This paper introduced GMEA, the first universal black-box framework for assessing the robustness of 3DGS watermarking systems. By formulating the attack as a group-based multi-objective optimization problem, GMEA effectively balances watermark removal with visual fidelity preservation, using a novel feature-variance objective to operate without detector knowledge. Our experiments show that GMEA effectively compromises both 1D and 2D watermarking schemes while maintaining high visual fidelity.

## References

Bao, Y.; Ding, T.; Huo, J.; Liu, Y.; Li, Y.; Li, W.; Gao, Y.; and Luo, J. 2025. 3D Gaussian Splatting: Survey, Technologies, Challenges, and Opportunities. IEEE Transactions on Circuits and Systems for Video Technology, 35(7): 6832â6852.

Chen, Z.; Wang, G.; Zhu, J.; Lai, J.; and Xie, X. 2025a. GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting. In Processings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 16325â16335.

Chen, Z.; Yang, J.; Huang, J.; de Lutio, R.; Esturo, J. M.; Ivanovic, B.; Litany, O.; Gojcic, Z.; Fidler, S.; Pavone, M.; Song, L.; and Wang, Y. 2025b. OmniRe: Omni Urban Scene Reconstruction. In International Conference on Learning Representations.

Cox, I. J.; Kilian, J.; Leighton, F. T.; and Shamoon, T. 1997. Secure spread spectrum watermarking for multimedia. IEEE Transactions on Image Processing, 6(12): 1673â1687.

Deb, K.; Pratap, A.; Agarwal, S.; and Meyarivan, T. 2002. A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2): 182â197.

Guedon, A.; and Lepetit, V. 2024. SuGaR: Surface-Aligned Â´ Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering. In Processings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 5354â5363.

Hong, H.; Jiang, M.; and Yen, G. G. 2023. Improving performance insensitivity of large-scale multiobjective optimization via Monte Carlo tree search. IEEE Transactions on Cybernetics, 54(3): 1816â1827.

Huang, X.; Li, R.; Cheung, Y.-m.; Cheung, K. C.; See, S.; and Wan, R. 2025. Gaussianmarker: Uncertainty-aware copyright protection of 3d gaussian splatting. In Processings of Neural Information Processing Systems.

Jang, Y.; Park, H.; Yang, F.; Ko, H.; Choo, E.; and Kim, S. 2025. 3D-GSW: 3D Gaussian Splatting for Robust Watermarking. arXiv:2409.13222.

Kerbl, B.; Kopanas, G.; Leimkhler, T.; and Drettakis, G. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics, 42(4).

Liang, J.; Ban, X.; Yu, K.; Qu, B.; Qiao, K.; Yue, C.; Chen, K.; and Tan, K. C. 2023. A Survey on Evolutionary Constrained Multiobjective Optimization. IEEE Transactions on Evolutionary Computation, 27(2): 201â221.

Lu, Y.; Jia, Y.; Wang, J.; Li, B.; Chai, W.; Carin, L.; and Velipasalar, S. 2020. Enhancing cross-task black-box transferability of adversarial examples with dispersion reduction. In Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition, 940â949.

MacQueen, J. 1967. Some methods for classification and analysis of multivariate observations.

Mildenhall, B.; Srinivasan, P. P.; Ortiz-Cayon, R.; Kalantari, N. K.; Ramamoorthi, R.; Ng, R.; and Kar, A. 2019. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics, 38(4): 1â14.

Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.; Ramamoorthi, R.; and Ng, R. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1): 99â106.

Papernot, N.; McDaniel, P.; Goodfellow, I.; Jha, S.; Celik, Z. B.; and Swami, A. 2017. Practical black-box attacks against machine learning. In ACM on Asia Conference on Computer and Communications Security, 506â519.

Setiadi, D. R. I. M. 2021. PSNR vs SSIM: imperceptibility quality assessment for image steganography. Multimedia Tools and Applications, 80(6): 8423â8444.

Tu, X.; Radl, L.; Steiner, M.; Steinberger, M.; Kerbl, B.; and de la Torre, F. 2025. VRSplat: Fast and Robust Gaussian Splatting for Virtual Reality. Processings of the ACM on Computer Graphics and Interactive Techniques, 8(1).

Van Gansbeke, W.; Vandenhende, S.; Georgoulis, S.; Proesmans, M.; and Van Gool, L. 2020. Scan: Learning to classify images without labels. In Processings of European Conference on Computer Vision, 268â285. Springer.

Wang, J.; Wang, Y.; Wang, H.; and Zhang, J. 2021. A survey on evolutionary computation for adversarial machine learning. IEEE Transactions on Evolutionary Computation, 26(5): 994â1009.

Wang, Z.; Zeng, Q.; Lin, W.; Jiang, M.; and Tan, K. C. 2024. Generating Diagnostic and Actionable Explanations for Fair Graph Neural Networks. In Processings of the AAAI Conference on Artificial Intelligence, volume 38, 21690â21698.

Wu, G.; Yi, T.; Fang, J.; Xie, L.; Zhang, X.; Wei, W.; Liu, W.; Tian, Q.; and Wang, X. 2024a. 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering. In Processings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 20310â20320.

Wu, T.; Yuan, Y.-J.; Zhang, L.-X.; Yang, J.; Cao, Y.-P.; Yan, L.-Q.; and Gao, L. 2024b. Recent advances in 3D Gaussian splatting. Computational Visual Media, 10(4): 613â642.

Xie, T.; Zong, Z.; Qiu, Y.; Li, X.; Feng, Y.; Yang, Y.; and Jiang, C. 2024. PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 4389â4398.

Yi, T.; Fang, J.; Wang, J.; Wu, G.; Xie, L.; Zhang, X.; Liu, W.; Tian, Q.; and Wang, X. 2024. GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 6796â6807.

Zeng, Q.; Wang, Z.; Cheung, Y.-m.; and Jiang, M. 2024. Ask, attend, attack: An effective decision-based black-box targeted attack for image-to-text models. In Processings of Neural Information Processing Systems, volume 37, 105819â105847.

Zhang, R.-Z.; Wang, S.-Z.; Liu, T.-C.; and Zhong, S.-M. 2020. A survey on adversarial attacks on deep-learning based steganography. IEEE Access, 8: 189518â189537.

Zhang, T.; Yu, H.-X.; Wu, R.; Feng, B. Y.; Zheng, C.; Snavely, N.; Wu, J.; and Freeman, W. T. 2024a. Physdreamer: Physics-based interaction with 3d objects via video

generation. In Processings of European Conference on Computer Vision, 388â406. Springer.

Zhang, X.; Meng, J.; Li, R.; Xu, Z.; Zhang, Y.; and Zhang, J. 2024b. GS-Hider: Hiding Messages into 3D Gaussian Splatting. In Globerson, A.; Mackey, L.; Belgrave, D.; Fan, A.; Paquet, U.; Tomczak, J.; and Zhang, C., eds., Processings of Neural Information Processing Systems, volume 37, 49780â 49805.

Zhang, X.; Tian, Y.; Cheng, R.; and Jin, Y. 2018. A Decision Variable Clustering-Based Evolutionary Algorithm for Large-Scale Many-Objective Optimization. IEEE Transactions on Evolutionary Computation, 22(1): 97â112.

Zhao, X.; Zhang, K.; Su, Z.; Vasan, S.; Grishchenko, I.; Kruegel, C.; Vigna, G.; Wang, Y.-X.; and Li, L. 2024. Invisible image watermarks are provably removable using generative ai. In Processings of Neural Information Processing Systems, volume 37, 8643â8672.

Zhu, R.; Liang, Y.; Chang, H.; Deng, J.; Lu, J.; Yang, W.; Zhang, T.; and Zhang, Y. 2025. MotionGS: exploring explicit motion guidance for deformable 3D Gaussian splatting. In Processings of Neural Information Processing Systems. Red Hook, NY, USA. ISBN 9798331314385.

Zhu, S.; Wang, G.; Kong, X.; Kong, D.; and Wang, H. 2024. 3D Gaussian Splatting in Robotics: A Survey. arXiv:2410.12262.

## Appendix

## A Overview

This appendix provides supplementary material to support and elaborate on the findings presented in our main paper, âFading the Digital Ink: A Universal Black-Box Attack Framework for 3DGS Watermarking Systems.â The goal is to offer greater detail and transparency regarding our methodology, implementation, and results. The structure of this appendix is as follows:

â¢ Reproducibility Details: A comprehensive guide to reproducing our work. This includes the detailed experimental settings (hyperparameters, hardware, software, watermark specifications, and evaluation metrics) and the formal pseudocode for our GMEA framework and its sub-components.

â¢ Theoretical Justification: The complete mathematical proof for our watermark destruction objective (F2), which formally establishes the relationship between minimizing feature variance and erasing information content.

â¢ Supplementary Experiments: A collection of additional results that were omitted from the main paper due to space constraints. This includes extensive qualitative comparisons on multiple datasets, a visual analysis of 2D watermark corruption, a full quantitative breakdown of visual fidelity across all attack variations, and an in-depth ablation study on the impact of population size.

â¢ Discussion: A broader discussion of our work, including an acknowledgment of the current limitations (such as computational cost) and an outline of promising directions for future research.

We believe these materials offer a comprehensive resource for readers interested in the technical details and reproducibility of our work.

Table 4: Visual fidelity of the attacked 3DGS models (1D watermark, Blender dataset) compared to the original watermarked models. Our full GMEA method maintains high visual quality, as shown by the high SSIM and PSNR, and low RMSE. The arrows indicate the desired direction for each metric.
<table><tr><td>Scene</td><td>SSIM â</td><td>PSNR â</td><td>RMSE â</td></tr><tr><td>Chair</td><td>95.48</td><td>29.05</td><td>0.0363</td></tr><tr><td>Drums</td><td>94.20</td><td>26.77</td><td>0.0489</td></tr><tr><td>Ficus</td><td>97.25</td><td>32.73</td><td>0.0258</td></tr><tr><td>Hotdog</td><td>96.82</td><td>33.27</td><td>0.0242</td></tr><tr><td>Lego</td><td>94.74</td><td>30.15</td><td>0.0347</td></tr><tr><td>Materials</td><td>95.57</td><td>30.92</td><td>0.0317</td></tr><tr><td>Mic</td><td>96.87</td><td>32.39</td><td>0.0263</td></tr><tr><td>Ship</td><td>90.13</td><td>30.00</td><td>0.0342</td></tr><tr><td>Average</td><td>95.13</td><td>30.66</td><td>0.0328</td></tr></table>

## B Reproducibility

Our source code and data are included in the supplemental material, and we will publish the code on GitHub after the paper is accepted to ensure full reproducibility.

## Experimental Settings

To ensure the transparency and reproducibility of our work, this section provides a comprehensive overview of the experimental environment and the hyperparameter configurations used for our GMEA framework. To ensure statistical reliability and account for the stochastic nature of the evolutionary algorithm, all reported quantitative results represent the average of 5 independent runs, each with a different random seed.

Watermark Details To ensure our experiments are fully reproducible, this section details the specific 1D and 2D watermarks used. For all experiments involving 1D watermarking (targeting the GaussianMarker method), we used a fixed 48-bit binary string as the copyright message to ensure consistency across different scenes and attack evaluations. The specific bitstream embedded was:

$$
\begin{array} { r } { \begin{array} { r r } { 1 1 1 0 1 0 1 1 , } \\ { 0 1 0 0 1 1 0 1 , } \end{array} \begin{array} { r } { 0 1 0 1 0 0 0 0 , 0 1 0 1 0 1 1 1 , } \\ { 0 1 0 0 0 1 0 0 , } \end{array} \begin{array} { r } { 0 0 1 0 0 1 1 1 1 , } \\ { 0 0 1 0 0 1 1 1 1 } \end{array} } \end{array}
$$

For the 2D watermarking experiments (targeting the GS-Hider method), the AAAI logo served as the embedded image watermark, as visualized in Figure 9.

Evaluation Metrics We assess our frameworkâs performance based on two primary criteria: the visual fidelity of the attacked 3DGS model and the efficacy of watermark removal. Visual quality is quantified using standard image metrics: PSNR, SSIM(%), and MSE, where higher PSNR and SSIM values, alongside a lower MSE, indicate better preservation of visual fidelity. For 1D bitstream watermarks, we measure removal efficacy with three metrics. The first is the standard Bit Accuracy Rate (BAR), or the proportion of correctly decoded bits, where 0.5 indicates a random output:

$$
\mathrm { \bf B A R } = \frac { \mathrm { N u m b e r ~ o f ~ C o r r e c t l y ~ D e c o d e d ~ B i t s } } { \mathrm { T o t a l ~ N u m b e r ~ o f ~ B i t s } }\tag{14}
$$

From this, we derive the Watermark Uncertainty Score (WUS), which measures closeness to random noise (a score of 1 is a perfect attack):

$$
\mathrm { W U S } = 1 - 2 \times | \mathrm { B A R } - 0 . 5 |\tag{15}
$$

Finally, to account for bit imbalances, we use the Information Destruction Score (IDS), derived from the Matthews Correlation Coefficient (MCC). An IDS of 1 signifies that the statistical correlation between the original and extracted watermarks is completely eliminated:

$$
\mathrm { I D S } = 1 - \left| { \frac { \mathrm { T P } \times \mathrm { T N } - \mathrm { F P } \times \mathrm { F N } } { \sqrt { ( \mathrm { T P } + \mathrm { F P } ) ( \mathrm { T P } + \mathrm { F N } ) ( \mathrm { T N } + \mathrm { F P } ) ( \mathrm { T N } + \mathrm { F N } ) } } } \right.\tag{16}
$$

where TP, TN, FP, and FN are the counts of True Positives, True Negatives, False Positives, and False Negatives for the decoded bits, respectively.

<!-- image-->  
Figure 6: Qualitative comparison on the Blender dataset. Each pair displays the rendered image from the original watermarked 3DGS model (left) alongside the render from the model after being processed by our GMEA framework (right). The high degree of visual similarity demonstrates that our attack successfully removes the watermark while preserving excellent visual fidelity, making the modifications virtually imperceptible.

<!-- image-->  
Figure 7: Convergence of Objective 1 (Quality Loss) vs. Generation. This plot shows that larger population sizes (e.g., NIND=70) achieve a lower (better) fitness value over 200 generations. However, this view does not account for the differing time costs.

<!-- image-->  
Figure 8: Convergence of Objective 2 (Watermark Destruction) vs. Generation. Similar to F1, larger populations show better convergence per generation, finding solutions that more effectively destroy the watermark by the end of the run.

Group-Based Strategy Settings The partitioning of the 3DGS model was controlled by the following parameter:

â¢ Number of Groups (k): For our main experiments (Tation in computation time while keeping the peak memory usage manageable.

<!-- image-->  
Figure 9: The 2D image watermark (AAAI logo) used for all experiments targeting the GS-Hider watermarking system.

Table 5: Visual fidelity of the attacked 3DGS models using the single-objective GMEA (w/o F1) on the 1D watermarked Blender dataset. As expected, removing the visual quality objective (F1) leads to a significant degradation in fidelity, reflected in the lower SSIM/PSNR and higher RMSE values. The arrows indicate the desired direction for each metric.
<table><tr><td>Scene</td><td>SSIM â</td><td>PSNR â</td><td>RMSE â</td></tr><tr><td>Chair</td><td>79.31</td><td>19.12</td><td>0.1100</td></tr><tr><td>Drums</td><td>83.24</td><td>20.16</td><td>0.0990</td></tr><tr><td>Ficus</td><td>88.52</td><td>22.59</td><td>0.0747</td></tr><tr><td>Hotdog</td><td>85.50</td><td>22.33</td><td>0.0784</td></tr><tr><td>Lego</td><td>78.73</td><td>21.12</td><td>0.0899</td></tr><tr><td>Materials</td><td>84.61</td><td>22.26</td><td>0.0784</td></tr><tr><td>Mic</td><td>90.53</td><td>25.19</td><td>0.0554</td></tr><tr><td>Ship</td><td>74.12</td><td>21.47</td><td>0.0855</td></tr><tr><td>Average</td><td>83.07</td><td>21.78</td><td>0.0839</td></tr></table>

ble 2 in main paper), we set the number of groups to k = 10. This value was chosen based on our analysis (Table 3 in main paper), as it offers a significant reduc-

Table 6: Visual fidelity of the attacked 3DGS models (2D watermark, LLFF dataset) compared to the original watermarked models. Our full GMEA method successfully preserves high visual quality on these complex scenes, as indicated by the excellent metrics. The arrows indicate the desired direction for each metric.
<table><tr><td>Scene</td><td>SSIM â</td><td>PSNR â</td><td>RMSE â</td></tr><tr><td>Fern</td><td>97.42</td><td>36.60</td><td>0.0148</td></tr><tr><td>Flower</td><td>98.81</td><td>40.64</td><td>0.0094</td></tr><tr><td>Fortress</td><td>98.73</td><td>39.95</td><td>0.0110</td></tr><tr><td>Horns</td><td>98.57</td><td>39.05</td><td>0.0115</td></tr><tr><td>Leaves</td><td>98.31</td><td>35.03</td><td>0.0178</td></tr><tr><td>Trex</td><td>96.47</td><td>32.00</td><td>0.0252</td></tr><tr><td>Room</td><td>98.64</td><td>39.60</td><td>0.0108</td></tr><tr><td>Orchids</td><td>98.80</td><td>39.24</td><td>0.0110</td></tr><tr><td>Average</td><td>98.22</td><td>37.76</td><td>0.0140</td></tr></table>

Table 7: Visual fidelity of the attacked 3DGS models using the single-objective GMEA (w/o F1) on the 1D watermarked LLFF dataset. This ablation study demonstrates a substantial degradation in visual quality, with very low SSIM/PSNR scores and high RMSE, confirming the necessity of the visual fidelity objective $( F _ { 1 } )$ . The arrows indicate the desired direction for each metric.
<table><tr><td>Scene</td><td>SSIM â</td><td>PSNR â</td><td>RMSE â</td></tr><tr><td>Fern</td><td>50.01</td><td>19.12</td><td>0.1106</td></tr><tr><td>Flower</td><td>48.70</td><td>19.31</td><td>0.1083</td></tr><tr><td>Fortress</td><td>56.58</td><td>19.81</td><td>0.1024</td></tr><tr><td>Horns</td><td>53.87</td><td>19.01</td><td>0.1123</td></tr><tr><td>Leaves</td><td>49.33</td><td>15.92</td><td>0.1602</td></tr><tr><td>Trex</td><td>43.71</td><td>16.72</td><td>0.1460</td></tr><tr><td>Room</td><td>65.70</td><td>20.81</td><td>0.0911</td></tr><tr><td>Orchids</td><td>54.79</td><td>18.85</td><td>0.1143</td></tr><tr><td>Average</td><td>52.84</td><td>18.69</td><td>0.1182</td></tr></table>

Evolutionary Algorithm Settings Our attack is driven by a multi-objective evolutionary algorithm. Its parameters were set as follows:

â¢ Population Size $( N _ { p o p } ) \colon$ We used a population of $N _ { p o p } = 5 0$ individuals for each sub-problemâs evolutionary run.

â¢ Number of Generations (T ): The optimization was executed for a total of $T ~ = ~ 2 0 0$ generations, which we observed was sufficient for the objective values to converge.

â¢ Color Perturbation (Ïµ): The range for the color perturbation vector $\mathbf { c } ^ { ( i ) }$ was bounded by $[ - \epsilon , \epsilon ] .$ , with $\epsilon =$ 50/255.

â¢ Crossover Operator: We used a simulated binary crossover (SBX) operator with a distribution index of

ssim: 50.84 psnr: 4.81  
ssim: 58.85 psnr: 5.34  
<!-- image-->

<!-- image-->  
ssim: 71.43 psnr: 8.68  
ssim: 76.15 psnr: 9.72

ssim: 63.96 psnr: 7.11  
<!-- image-->  
ssim: 85.3 psnr: 14.21

Figure 10: Qualitative results of 2D watermark extraction after the GMEA attack. The six examples illustrate the varying degrees of watermark corruption achieved by different solutions on the Pareto front, ranging from minimally distorted (corresponding to solutions that prioritize the 3DGS modelâs visual quality) to almost completely destroyed (corresponding to solutions that prioritize attack efficacy).

$$
\eta _ { c } = 1 . 0 .
$$

â¢ Mutation Operator: A polynomial mutation operator was applied with a mutation probability of $p _ { m } ~ = ~ 0 . 1$ and a distribution index of $\eta _ { m } = 2 0$

Objective Function Settings The two competing objectives were configured with these parameters:

â¢ Visual Fidelity (F1): The weight Î» in Equation 7 (main paper), which balances the L1 and SSIM losses, was set to Î» = 0.85. The fitness evaluation was performed over a batch of $N _ { v } = 8$ randomly sampled camera views in each generation.

â¢ Watermark Destruction (F2): The feature extractor Î¦ (Equation 9 in main paper) was a pre-trained VGG-19 network. We extracted features from the output of the ârelu4 1â layer, as its mid-level representational power is well-suited for capturing the patterns that constitute a watermark.

Hardware and Software Environment All experiments were conducted on a high-performance computing server with the following specifications:

â¢ CPU: Intel(R) Xeon(R) Gold 5222 CPU @ 3.80GHz

â¢ GPU: 2x NVIDIA A40 (48 GB VRAM each)

â¢ Operating System: A Linux-based distribution

â¢ Core Libraries: PyTorch was used for deep learning operations, including model rendering and feature extraction. The multi-objective evolutionary optimization was implemented using a standard Python library for evolutionary computation.

## Explanation of Algorithm 1: GMEA Framework

To provide a clear and formal description of our attack framework, we present the corresponding pseudocode below. Our methodology consists of two main algorithms. Algorithm 1 outlines the high-level strategy of the Groupbased Multi-objective Evolutionary Attack (GMEA), which partitions the problem and orchestrates the overall attack. Algorithm 2 details the core optimization process, an evolutionary attack applied to each sub-model, which is called within Algorithm 1. All the equations involved are those in the main paper.

Algorithm 1 describes the main workflow of our proposed GMEA.

â¢ Step 1 (Line 3): Partitioning Phase. The process begins by partitioning the entire 3DGS model, which can contain millions of Gaussians, into k smaller, computationally manageable sub-models using K-Means. This step is crucial for making the large-scale optimization problem tractable.

â¢ Step 2 (Lines 4-6): Divide-and-Conquer. The core of our framework is a divide-and-conquer strategy. We iterate through each sub-model and apply an independent optimization process, EvolveSubModel (detailed in Algorithm 2), to it. This design allows for massive parallelization, significantly speeding up the attack.

â¢ Step 3 (Line 7): Reconstruction. Finally, after each sub  
model has been optimized, they are merged back together   
to reconstruct the final adversarial 3DGS model, $\bar { \mathbf { G } } _ { a d v }$   
Algorithm 1: Group-based Multi-objective Evolutionary At  
tack (GMEA)   
1: Input: Watermarked 3DGS model $\mathbf { G } _ { w m }$ , number of   
clusters k.   
2: Output: Adversarial 3DGS model $\mathbf { G } _ { a d v } .$   
3: Partition $\mathbf { G } _ { w m }$ into k sub-models $\{ \mathbf { G } _ { w m } ^ { ( 1 ) } , \ldots , \mathbf { G } _ { w m } ^ { ( k ) } \}$   
using K-Means clustering with Eq. (2), (3), and (4).   
4: for all sub-model $\mathbf { G } _ { w m } ^ { ( i ) }$ in $\{ \mathbf { G } _ { w m } ^ { ( 1 ) } , \dotsc , \mathbf { G } _ { w m } ^ { ( k ) } \}$ do   
5: $\mathbf { G } _ { a d v } ^ { ( i ) } $ EvolveSubMode $| ( \mathbf { G } _ { w m } ^ { ( i ) } )$   
6: end for   
7: Reconstruct the final model ${ \bf G } _ { a d v }$ by merging all opti  
mized sub-models using Eq. (13).   
8: return $\mathbf { G } _ { a d v }$

## Explanation of Algorithm 2: Evolutionary Attack on a Sub-Model

Algorithm 2 provides the implementation details for the EvolveSubModel function, which is the heart of our attack mechanism. Its goal is to find the optimal perturbation for a given sub-model to erase the watermark while preserving visual fidelity. All the equations involved are those in the main paper.

â¢ Step 1 (Line 3): Population Initialization. We initialize a population Pt, where each individual x represents a potential set of modifications to the attributes (e.g., color, opacity, scale) of the Gaussians in the sub-model.

â¢ Step 2 (Lines 4-12): Evolutionary Loop. The algorithm enters the main evolutionary loop. In each generation, we create new candidate solutions (offspring) using crossover and mutation operators.

â¢ Step 3 (Lines 7-10): Fitness Calculation. Each candidate solution is evaluated using a fitness function with two competing objectives: $F _ { 1 }$ , which measures the success of watermark removal, and $F _ { 2 } ,$ which quantifies the visual similarity to the original model.

â¢ Step 4 (Line 11): Environmental Selection. A selection mechanism based on non-dominated sorting and a density metric (as used in NSGA-II) is employed to choose the individuals that will form the next generationâs population. This preserves a diverse set of high-quality solutions that balance the two objectives.

â¢ Step 5 (Lines 13-14): Final Solution Selection. After the final generation, the best solution xâ from the Pareto front is selected to construct the final optimized submodel $\mathbf { G } _ { a d v } ^ { ( i ) }$ â¢

Algorithm 2: Evolutionary Attack on a Sub-Model   
1: Input: A watermarked 3DGS sub-model $\mathbf { G } _ { w m } ^ { ( i ) }$   
2: Output: An optimized adversarial sub-model $\mathbf { G } _ { a d v } ^ { ( i ) } .$   
3: Initialize population $\mathcal { P } _ { t }$ with individuals x defined by   
Eq. (5).   
4: for $t = 1$ to $T$ do   
5: Generate offspring $\mathcal { Q } _ { t }$ from $\mathcal { P } _ { t }$ using crossover (Eq.   
10) and mutation (Eq. 11).   
6: Combine populations into a pool $\mathcal { R } _ { t } = \mathcal { P } _ { t } \cup \mathcal { Q } _ { t }$   
7: for all individual x in $\mathcal { R } _ { t }$ do   
8: Construct candidate $\mathbf { G } ^ { \prime }$ from x via Eq. (6).   
9: Calculate fitness $( F _ { 1 } , F _ { 2 } )$ using Eq. (7), (8), (9).   
10: end for   
11: Select next generation $\mathcal { P } _ { t + 1 }$ via non-dominated sort  
ing and density metric (Eq. 12).   
12: end for   
13: Select best solution $\mathbf { x } ^ { * }$ from the final population $\mathcal { P } _ { t + 1 }$   
14: Construct final $\mathbf { G } _ { a d v } ^ { ( i ) }$ using the best solution $\mathbf { x } ^ { * } .$   
15: return $\mathbf { G } _ { a d v } ^ { ( i ) } .$

## C Theoretical Justification

To provide a theoretical foundation for our watermark destruction objective $F _ { 2 }$ , we introduce and prove Lemma 1. This lemma establishes a direct relationship between the variance of feature map activations and their information content. We model the activation values within a feature channel as a continuous random variable Z.

Lemma 1. For a continuous random variable with a given mean and variance, its differential entropy is upper-bounded by the entropy of a Gaussian distribution with the same mean and variance. This upper bound is a strictly monotonically increasing function of the variance. Consequently, minimizing the variance of the random variable forces a compression of its information entropyâs upper bound, causing its distribution to degenerate into an uninformative Dirac delta function in the limit.

<!-- image-->  
Figure 11: Visualization of our group-based partitioning strategy with k = 5 on scenes from the Blender dataset. The leftmost column shows the render of the original, complete 3DGS model. The subsequent five columns display the individual sub-3DGS models, each representing a distinct and spatially coherent sub-problem for optimization.

<!-- image-->  
Figure 12: Qualitative comparison on the more complex LLFF dataset. Each pair consists of the original watermarked 3DGS render (left) and the unwatermarked version produced by our GMEA attack (right). Even in these challenging real-world scenes, the attacked renders are nearly indistinguishable from the originals, underscoring the effectiveness and robustness of our method in maintaining visual quality.

Table 8: Visual fidelity of the attacked 3DGS models using the single-objective GMEA (w/o F1) on the 2D watermarked Blender dataset. The significant drop in SSIM/P-SNR and the high RMSE values demonstrate the severe visual degradation when the fidelity-preserving objective $( F _ { 1 } )$ is omitted. The arrows indicate the desired direction for each metric.
<table><tr><td>Scene</td><td>SSIM â</td><td>PSNR â</td><td>RMSE â</td></tr><tr><td>Chair</td><td>85.54</td><td>18.50</td><td>0.1207</td></tr><tr><td>Drums</td><td>79.34</td><td>18.09</td><td>0.1251</td></tr><tr><td>Ficus</td><td>86.09</td><td>19.64</td><td>0.1051</td></tr><tr><td>Hotdog</td><td>86.50</td><td>21.90</td><td>0.0827</td></tr><tr><td>Lego</td><td>72.87</td><td>17.20</td><td>0.1408</td></tr><tr><td>Materials</td><td>81.30</td><td>20.40</td><td>0.0971</td></tr><tr><td>Mic</td><td>86.27</td><td>21.18</td><td>0.2811</td></tr><tr><td>Ship</td><td>70.93</td><td>20.57</td><td>0.0945</td></tr><tr><td>Average</td><td>81.11</td><td>19.69</td><td>0.1309</td></tr></table>

<!-- image-->  
Figure 13: Convergence of Objective 1 (Quality Loss) vs. Time (Sec.). This plot provides a fairer efficiency comparison. It reveals that while larger populations eventually find better solutions, a smaller population (e.g., NIND=30 or 50) can reach a good-quality solution much faster.

We divide this proof into four parts.

A. The Gaussian Distribution Maximizes Entropy. We first prove that among all continuous probability distributions with a given mean $\mu$ and variance $\sigma ^ { 2 }$ , the Gaussian distribution $\bar { \mathcal { N } } ( \mu , \sigma ^ { 2 } )$ has the maximum differential entropy. Let $p ( z )$ be an arbitrary probability density function (PDF) with mean $\mu$ and variance $\sigma ^ { 2 }$ , and let $g ( z )$ be the PDF of a Gaussian distribution with the same parameters. We consider the Kullback-Leibler (KL) divergence between these two distributions:

$$
D _ { K L } ( p | | g ) = \int _ { - \infty } ^ { \infty } p ( z ) \log \frac { p ( z ) } { g ( z ) } d z\tag{17}
$$

By Gibbsâ inequality, we know that $D _ { K L } ( p | | g ) \geq 0$ , with equality holding if and only $\mathrm { i f } p ( z ) = g ( z )$ . Expanding the

definition of KL divergence:

$$
D _ { K L } ( p | | g ) = \int _ { - \infty } ^ { \infty } p ( z ) \log p ( z ) d z - \int _ { - \infty } ^ { \infty } p ( z ) \log g ( z ) d z\tag{18}
$$

The first term is the negative differential entropy of $p ( z )$ , i.e., $- H ( p )$ . For the second term, the logarithm of the Gaussian PDF $g ( z )$ is:

$$
\begin{array} { c } { { \log g ( z ) = \log \left( \frac 1 { \sqrt { 2 \pi \sigma ^ { 2 } } } e ^ { - \frac { ( z - \mu ) ^ { 2 } } { 2 \sigma ^ { 2 } } } \right) } } \\ { { = - \frac { ( z - \mu ) ^ { 2 } } { 2 \sigma ^ { 2 } } - \log ( \sqrt { 2 \pi \sigma ^ { 2 } } ) } } \end{array}\tag{19}
$$

Substituting this into the integral of the second term:

$$
\int _ { - \infty } ^ { \infty } p ( z ) \log g ( z ) d z\tag{20}
$$

$$
= \int _ { - \infty } ^ { \infty } p ( z ) \left( - { \frac { ( z - \mu ) ^ { 2 } } { 2 \sigma ^ { 2 } } } - \log ( { \sqrt { 2 \pi \sigma ^ { 2 } } } ) \right) d z\tag{21}
$$

$$
= - { \frac { 1 } { 2 \sigma ^ { 2 } } } \int _ { - \infty } ^ { \infty } p ( z ) ( z - \mu ) ^ { 2 } d z\tag{22}
$$

$$
- \log ( \sqrt { 2 \pi \sigma ^ { 2 } } ) \int _ { - \infty } ^ { \infty } p ( z ) d z\tag{23}
$$

By definition, $\begin{array} { r } { \int p ( z ) ( z - \mu ) ^ { 2 } d z = \sigma ^ { 2 } } \end{array}$ (the variance of $p ( z ) )$ and $\textstyle \int p ( z ) d z = 1$ . This simplifies the expression to:

$$
- \frac { 1 } { 2 \sigma ^ { 2 } } ( \sigma ^ { 2 } ) - \log ( \sqrt { 2 \pi \sigma ^ { 2 } } )\tag{24}
$$

$$
= - \frac { 1 } { 2 } - \frac { 1 } { 2 } \log ( 2 \pi \sigma ^ { 2 } )\tag{25}
$$

$$
= - \frac { 1 } { 2 } \log ( 2 \pi e \sigma ^ { 2 } )\tag{26}
$$

This is precisely the negative differential entropy of the Gaussian distribution, $- H ( g )$ . The KL divergence thus becomes:

DKL(p||g) = âH(p) â (âH(g)) = H(g) â H(p) (27) Since $D _ { K L } ( p | | g ) \geq 0 .$ , it follows that $H ( g ) - H ( p ) \geq 0 .$ which implies $H ( p ) \leq H ( g )$ . This proves that the entropy of a Gaussian distribution is the upper bound for any distribution with the same variance. Thus, $H _ { \mathrm { m a x } } ( Z ) = H ( g )$

B. Monotonicity of the Entropy Bound w.r.t. Variance. Having established the entropy upper bound as $H _ { \operatorname* { m a x } } ( Z ) =$ $\scriptstyle { \frac { 1 } { 2 } } \log ( 2 \pi e \sigma ^ { 2 } )$ , we now show its monotonicity with respect to the variance $\sigma ^ { 2 }$ . We find the derivative of $H _ { \mathrm { m a x } } ( Z )$ with respect to $\sigma ^ { 2 }$ :

$$
\frac { d H _ { \mathrm { m a x } } ( Z ) } { d ( \sigma ^ { 2 } ) } = \frac { d } { d ( \sigma ^ { 2 } ) } \left( \frac { 1 } { 2 } \log ( 2 \pi e \sigma ^ { 2 } ) \right)\tag{28}
$$

$$
= { \frac { 1 } { 2 } } \cdot { \frac { 1 } { 2 \pi e \sigma ^ { 2 } } } \cdot ( 2 \pi e )\tag{29}
$$

$$
= { \frac { 1 } { 2 \sigma ^ { 2 } } }\tag{30}
$$

Since the variance $\sigma ^ { 2 }$ is strictly positive for any nondegenerate distribution $( \sigma ^ { 2 } > 0 )$ , the derivative $\scriptstyle { \frac { 1 } { 2 \sigma ^ { 2 } } }$ is always positive. Therefore, $H _ { \mathrm { m a x } } ( Z )$ is a strictly monotonically increasing function of the variance $\sigma ^ { 2 }$

C. Degeneracy to the Dirac Delta Function in the Limit. Next, we analyze the behavior of the Gaussian distribution ${ \mathcal { N } } ( \mu , \sigma ^ { 2 } )$ as its variance approaches zero, i.e., $\sigma ^ { 2 } \to 0 ,$ . The PDF is $\begin{array} { r } { g ( z ; \mu , \sigma ) = \frac { 1 } { \sqrt { 2 \pi } \sigma } e ^ { - \frac { ( z - \mu ) ^ { 2 } } { 2 \sigma ^ { 2 } } } } \end{array}$ . We examine its properties in the limit $\sigma \to 0 ^ { + }$

1. For $z ~ \neq ~ \mu :$ The term $( z \ - \ \mu ) ^ { 2 } > 0 .$ . As $\sigma $ $0 ^ { + }$ , the exponent $\begin{array} { r l r } { - \frac { ( z - \mu ) ^ { 2 } } { 2 \sigma ^ { 2 } } } & { { } \to } & { - \infty } \end{array}$ . Consequently, $\begin{array} { r } { \operatorname* { l i m } _ { \sigma \to 0 ^ { + } } g ( z ; \mu , \sigma ) = 0 \mathrm { ~ \bar { f } o r ~ a l l ~ } z \neq \mu . } \end{array}$

2. For $z = \mu \colon$ The exponent is 0, making $e ^ { 0 } = 1$ . However, the leading coefficient $\scriptstyle { \frac { 1 } { \sqrt { 2 \pi } \sigma } } \to$ +â as $\sigma  0 ^ { + }$ . Thus, lim ${ \operatorname { 1 } } _ { \sigma \to 0 ^ { + } } g ( \mu ; \mu , \sigma ) = \infty$

3. Integral Property: For any $\sigma > 0$ , the total integral of the PDF is unity: $\begin{array} { r } { \int _ { - \infty } ^ { \infty } g ( z ; \mu , \sigma ) d z = 1 } \end{array}$ . This property holds even in the limit.

Collectively, these three propertiesâbeing zero everywhere except at a single point where it is infinite, while maintaining a total integral of oneâare the defining characteristics of the Dirac delta function, $\delta ( z - \mu )$ . Thus, we formally deduce that lim $\mathfrak { i } _ { \sigma \to 0 ^ { + } } \mathcal { N } ( \mu , \sigma ^ { 2 } ) = \delta ( z - \mu )$

Table 9: Visual fidelity of the attacked 3DGS models using the single-objective GMEA (w/o F1) on the 2D watermarked LLFF dataset. The extremely low SSIM/PSNR scores and high RMSE values show that this attack variant severely compromises the visual integrity of the models, making the output unusable. The arrows indicate the desired direction for each metric.
<table><tr><td>Scene</td><td>SSIM â</td><td>PSNR â</td><td>RMSE â</td></tr><tr><td>Fern</td><td>47.56</td><td>17.13</td><td>0.1392</td></tr><tr><td>Flower</td><td>45.83</td><td>16.42</td><td>0.1512</td></tr><tr><td>Fortress</td><td>29.49</td><td>12.19</td><td>0.2459</td></tr><tr><td>Horns</td><td>53.55</td><td>16.89</td><td>0.1437</td></tr><tr><td>Leaves</td><td>29.49</td><td>12.23</td><td>0.2447</td></tr><tr><td>Trex</td><td>11.45</td><td>11.32</td><td>0.2718</td></tr><tr><td>Room</td><td>66.26</td><td>18.54</td><td>0.1186</td></tr><tr><td>Orchids</td><td>35.66</td><td>12.92</td><td>0.2260</td></tr><tr><td>Average</td><td>39.91</td><td>14.71</td><td>0.1926</td></tr></table>

D. Entropy in the Limit Case. Finally, we compute the limit of the entropy upper bound $H _ { \mathrm { m a x } } ( Z )$ as $\sigma \to 0 ^ { + }$

$$
\operatorname* { l i m } _ { \sigma \to 0 ^ { + } } H _ { \mathrm { m a x } } ( Z ) = \operatorname* { l i m } _ { \sigma \to 0 ^ { + } } \frac 1 2 \log ( 2 \pi e \sigma ^ { 2 } )\tag{31}
$$

As $\sigma  0 ^ { + }$ , the argument of the logarithm $2 \pi e \sigma ^ { 2 }  0 ^ { + }$ Since lim $\mathfrak { i } _ { x \to 0 ^ { + } } \log ( x ) = - \infty$ , it follows that:

$$
\operatorname* { l i m } _ { \sigma \to 0 ^ { + } } H _ { \mathrm { m a x } } ( Z ) = - \infty\tag{32}
$$

This result indicates that as the variance approaches zero, the information entropy of the distribution approaches negative infinity. This signifies a complete absence of uncertaintyâthe systemâs state becomes perfectly deterministic and thus contains no information.

This proof rigorously establishes that minimizing the variance of feature map activations, as proposed in our objective function $F _ { 2 }$ , directly compresses the upper bound of their information entropy. This process forces the feature representation to become uniform and predictable, thereby destroying any complex patterns, including the embedded watermark, that rely on informational diversity.

<!-- image-->  
Figure 14: Convergence of Objective 2 (Watermark Destruction) vs. Time (Sec.). This crucial comparison shows the time-based efficiency of watermark removal. It highlights that an intermediate population size offers the best trade-off, achieving significant watermark destruction in the shortest amount of time.

## D Supplementary Experiments

To complement the quantitative results in the main paper, we provide additional visual evidence and ablation studies that further validate the effectiveness and design choices of our GMEA framework.

## Qualitative Visual Results of GMEA

Visual quality is paramount for a successful attack, as the goal is to obtain a high-fidelity, unwatermarked asset. To this end, we present qualitative comparisons on both the Blender and LLFF datasets to demonstrate the imperceptibility of our full GMEA framework.

Figure 6 showcases the results on various object-centric scenes from the Blender dataset. Each image pair compares the original watermarked render with the render after our attack. The visual differences are minimal, confirming that our method successfully preserves the modelâs visual integrity while removing the watermark.

To demonstrate the robustness of our approach on more challenging data, Figure 12 presents the same comparison on complex, real-world scenes from the LLFF dataset. Even with intricate lighting and geometry, our GMEA framework maintains exceptional visual fidelity, rendering the attack practically invisible to the human eye. This visual evidence strongly supports the quantitative metrics presented in the main paper and in our ablation tables, underscoring the stealth and effectiveness of our method.

## Quantitative Analysis of Visual Fidelity

In addition to the visual comparisons, we provide a detailed quantitative analysis of the visual fidelity of the attacked models across different settings. This analysis, presented in the tables below, serves two purposes: first, to numerically confirm the high quality preserved by our full GMEA framework, and second, to provide a thorough ablation study justifying our multi-objective design.

Table 10: Computational cost analysis for different population sizes. The table shows the average wall-clock time required to compute a single generation for each population size (NIND) setting used in our experiments.
<table><tr><td>Population Size  $( N _ { p o p } )$ </td><td>Time per Generation (s)</td></tr><tr><td>70</td><td>5.5</td></tr><tr><td>50</td><td>4.5</td></tr><tr><td>30</td><td>2.6</td></tr><tr><td>10</td><td>0.9</td></tr></table>

We begin by evaluating our complete two-objective GMEA framework. Table 4 details the excellent visual quality metrics achieved when attacking 1D watermarked models on the Blender dataset. Similarly, Table 6 shows the results for attacking 2D watermarked models on the more complex LLFF dataset. In both tables, the high average SSIM (above 95%) and PSNR values, combined with low RMSE, provide strong quantitative evidence that our method preserves the assetâs original visual quality.

To validate our choice of a multi-objective approach, we conducted an ablation study by removing the visual fidelity objective $( F _ { 1 } )$ and running a single-objective attack (GMEA (w/o F1)). The results of this study are detailed in the subsequent four tables. Table 5 and Table 7 quantify the significant visual degradation when attacking 1D watermarks on the Blender and LLFF datasets, respectively. Likewise, Table 8 and Table 9 demonstrate a severe drop in quality for attacks on 2D watermarked models. Collectively, the low SSIM/PSNR scores and high RMSE values in these four tables confirm that a single-objective attack, while potent, renders the 3DGS asset visually unusable. This ablation study robustly justifies the necessity of our multi-objective formulation for any practical attack scenario where the goal is to steal a visually intact digital asset.

## Qualitative Analysis of 2D Watermark Corruption

In addition to evaluating the visual quality of the attacked 3DGS models, we also provide a qualitative analysis of the 2D watermarkâs degradation. Figure 10 showcases six examples of the extracted AAAI logo after our GMEA framework has been applied. These examples correspond to different optimal solutions found during the multi-objective optimization, each representing a unique trade-off between preserving the 3DGS modelâs visual fidelity $( F _ { 1 } )$ and maximizing watermark destruction $( F _ { 2 } )$ . The results visually confirm our attackâs effectiveness, demonstrating a spectrum of corruption from partially recognizable to almost completely erased. This supports the quantitative findings in the main paper, where lower SSIM/PSNR values for the extracted watermark indicate successful attacks.

## Visualization of the Group-Based Strategy

To provide an intuitive understanding of our group-based optimization strategy, we present visualizations of the partitioning process. Figure 11 and Figure 15 display the results of applying our strategy with $k = 5$ to scenes from the Blender and LLFF datasets, respectively. In each figure, the first column shows the fully rendered original 3DGS model. The following five columns show the individual renders of each of the five sub-3DGS models generated by our K-Means clustering algorithm. As can be seen, the strategy effectively decomposes the complex scene into spatially coherent and localized sub-problems (e.g., separating the shipâs sail from its hull, or isolating a specific cluster of flowers). This partitioning is the key to making the largescale search space manageable and improving the efficiency of the subsequent multi-objective optimization.

## Ablation Study on Population Size

To analyze the impact of the evolutionary algorithmâs population size $( N _ { p o p }$ or NIND) on performance, we conducted an ablation study comparing four different settings: 10, 30, 50, and 70. A key consideration is the trade-off between the quality of the solution found per generation and the wallclock time required to achieve it. We therefore present the convergence analysis from two distinct viewpoints.

First, we analyze convergence against the number of evolutionary steps. Figure 7 and Figure 8 show the performance of the two objective functions plotted against the generation number. These figures illustrate the raw evolutionary progress, where larger populations tend to explore the search space more effectively and achieve better fitness values at later generations.

Second, for a more practical assessment of efficiency, we analyze convergence against real-world time. This perspective is crucial because larger populations require significantly more computation per generation. We quantify this computational cost in Table 10, which lists the average wallclock time required to complete a single generation for each population size. Using this timing data, we re-plotted the convergence curves against time in seconds, as shown in Figure 13 and Figure 14. These plots provide a fair comparison of how quickly each configuration can reach a satisfactory solution and help answer which population size is most efficient within a given time budget.

## E Discussion

Our work introduces GMEA, the first universal black-box attack framework for 3DGS watermarking, demonstrating significant vulnerabilities in current copyright protection schemes. The success of our multi-objective, group-based evolutionary approach underscores the need for more robust watermarking techniques. In this section, we discuss the current limitations of our method and outline promising directions for future research.

## Limitations

While our group-based optimization strategy significantly accelerates the attack process, the framework still requires a moderate runtime. Depending on the modelâs complexity, a complete attack takes approximately 20 minutes to converge. This is due to the inherently iterative nature of the evolutionary algorithm, where each of the numerous fitness evaluations involves rendering the 3DGS model from multiple viewpoints and processing these images through a convolutional neural network.

original 3DGS  
sub-3DGS #1  
sub-3DGS #2  
sub-3DGS #3  
sub-3DGS #4  
sub-3DGS #5  
<!-- image-->  
Figure 15: Visualization of our group-based partitioning strategy with k = 5 on scenes from the more complex LLFF dataset. The first column shows the complete 3DGS model, while the following five columns show the spatially distinct sub-3DGS models produced by our clustering approach. This demonstrates the strategyâs effectiveness in segmenting complex real-world scenes.

However, it is crucial to contextualize this runtime. An attack on a digital asset is not a time-sensitive task, unlike applications such as real-time rendering. Given that the framework successfully removes the watermark in a challenging black-box setting while preserving high visual fidelity, we argue that this modest time cost is a highly acceptable tradeoff. For a malicious actor, the ability to obtain a high-quality, unwatermarked asset makes this time investment a negligible factor.

## Future Work

Building upon our findings, a key avenue for future research is the acceleration of the attack process to improve its efficiency. While not a critical limitation for the attackâs purpose, reducing the computational overhead would make the framework more accessible and faster to deploy. We propose several promising directions:

â¢ Surrogate-Assisted Optimization: The most significant bottleneck is the fitness evaluation step. Future work could explore the use of surrogate models (or proxy models), such as small, lightweight neural networks. These models could be trained to approximate the expensive objective functions $( F _ { 1 }$ and $\bar { F _ { 2 } ) }$ and would be used to pre-screen a large number of candidate solutions. The full, expensive evaluation would then be used only for the most promising individuals, drastically reducing the overall computational load.

â¢ Gradient Estimation in Black-Box Settings: Although we cannot access the true gradients of the watermark detector, techniques for gradient estimation in black-box settings could be explored. Methods like Natural Evolution Strategies (NES) or finite-difference approximations could provide an estimated gradient to guide the search more directly, potentially leading to much faster convergence than the derivative-free approach of our current evolutionary algorithm.

â¢ Enhanced Parallelization: Our group-based strategy already allows for a high degree of parallelization. This could be further enhanced by implementing a more sophisticated distributed computing framework, allowing the optimization of different sub-problems to scale across multiple machines in a network, rather than just multiple GPUs on a single server.

By pursuing these research directions, the efficiency of black-box attacks on 3DGS watermarking can be significantly improved, further highlighting the need for the development of more secure and robust copyright protection technologies.