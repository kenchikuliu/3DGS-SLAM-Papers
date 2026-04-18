# Thinking Like Van Gogh: Structure-Aware Style Transfer via Flow-Guided 3D Gaussian Splatting

Seeking âExaggeration in the Essentialâ through Geometric Abstraction

Zhendong Wangââ³, Lebin Zhouââ³, Jingchuan Xiaoâ â³, Rongduo Hanâ¡, Nam Lingâ, Cihan Ruanââ¦ zwang29@scu.edu, josephxky@gmail.com, lzhou@scu.edu, hrd12910@gmail.com, nling@scu.edu, cihanruan@ieee.org âDepartment of Computer Science and Engineering, Santa Clara University, Santa Clara, CA, USA â Department of Mathematics and Computer Studies, Mary Immaculate College, Limerick, Ireland â¡College of Software, Nankai University, Tianjin, China

<!-- image-->  
Fig. 1: Subjectivity over Physics. While the baseline method (middle) rigidly preserves photorealistic perspective and lightingâtreating style as a flat textureâour method (right) prioritizes subjective geometric flow. We demonstrate that authentic stylization requires sacrificing objective physical fidelity to reconstruct the expressive structural abstraction of the artist.

AbstractâIn 1888, Vincent van Gogh wrote, âI am seeking exaggeration in the essential.â This principleâamplifying structural form while suppressing photographic detailâlies at the core of Post-Impressionist art. However, most existing 3D style transfer methods invert this philosophy, treating geometry as a rigid substrate for surface-level texture projection. To authentically reproduce Post-Impressionist stylization, geometric abstraction must be embraced as the primary vehicle of expression.

We propose a flow-guided geometric advection framework for 3D Gaussian Splatting (3DGS) that operationalizes this principle in a mesh-free setting. Our method extracts directional flow fields from 2D paintings and back-propagates them into 3D space, rectifying Gaussian primitives to form flow-aligned brushstrokes that conform to scene topology without relying on explicit mesh priors. This enables expressive structural deformation driven directly by painterly motion rather than photometric constraints.

Our contributions are threefold: (1) a projection-based, meshfree flow guidance mechanism that transfers 2D artistic motion into 3D Gaussian geometry; (2) a luminanceâstructure decoupling strategy that isolates geometric deformation from color optimization, mitigating artifacts during aggressive structural abstraction; and (3) a VLM-as-a-Judge evaluation framework that assesses artistic authenticity through aesthetic judgment instead of conventional pixel-level metrics, explicitly addressing the subjective nature of artistic stylization.

Code: https://github.com/zhendong-zdw/TLVG-GS

Index Termsâ3D Gaussian Splatting, Neural Style Transfer, Flow-Guided Rendering, Geometric Stylization, Post-Impressionist Art, Non-Photorealistic Rendering, Van Gogh, Expressionism

## I. INTRODUCTION

N August 1888, Vincent van Gogh wrote from Arles: âI am seeking exaggeration in the essential.â [1] This radical declarationâborn from his study of Japanese ukiyo-e, where economy of line conveyed maximum expressionâwould catalyze modern artâs trajectory toward geometric abstraction [2]. Van Goghâs Arles period (1888â1890) enacted this principle: he abandoned fine academic brushwork for palette knives and loaded brushes, creating paint ridges up to 3mm thick [3] that encoded geometry through directional orientation, not photographic detail. Wheat fields dissolved into turbulent currents; stars became psychological spirals [4]. What critics derided as âcrudeâ proved revolutionaryâinspiring Post-Impressionism and Expressionism to embrace structural syntax over literal representation [5]. Each thick, directional stroke functions as a structural vector, encoding perceived 3D form through orientation alone (Fig. 1, left, Style Reference).

By systematically eliminating fine detail, Van Gogh compelled viewers to perceive volume through directional flow rather than photographic informationâwhat art historians describe as cross-contouring [6], where brushwork aligns with the principal curvature of an object to construct form. This insight, which catalyzed Post-Impressionism and Expressionism, reveals abstraction not as loss but as transformation: a trade from local fidelity toward global geometric coherence [7]. There is a recent computational analysis further confirms that Van Goghâs rhythmic, directionally coherent brushstrokes quantitatively distinguish his works from those of his contemporaries [8].

This artistic principle has direct computational implications.

<!-- image-->  
Fig. 2: Directional Syntax. Van Gogh (left): turbulent flow. Munch (right): laminar flow. Both prioritize geometric coherence.

Recent work demonstrated that parameterized brushstrokes, not pixel-level manipulation, capture authenticity in 2D [9]â [13]âconfirming that orientation is the syntax of style. Yet extending this to 3D remains unsolved. Existing neural style transfer methods, while achieving multi-view consistency [14], [15], fundamentally misunderstand Post-Impressionist painting. By treating style as statistical color distributions (Gram matrices) [16], while rigidly preserving geometric detail, they invert Van Goghâs principle: they preserve what he eliminated (photographic fidelity) and eliminate what he exaggerated (geometric flow) [17].

In this paper, we propose âThinking Like Van Goghâ, a flow-guided geometric advection framework for 3D Gaussian Splatting (3DGS) that realizes Post-Impressionist principles computationally. Our key insight: 2D directional flow fields in paintings encode the 3D structure artists perceived. By extracting these patterns and back-propagating them to rectify 3D Gaussians, we transform chaotic point clouds into coherent, flow-aligned brushstrokes that wrap around scene topology (Fig. 1, right). This âexaggeratesâ geometric structure while âeliminatingâ photographic detailâdeliberate abstraction as feature, not bug.

To achieve this, we introduce three technical innovations. First, we develop a mesh-free flow guidance algorithm that âcombsâ 3DGS primitives into directional brushstrokes via 2D-to-3D back-propagation, resolving the âfloating stickerâ artifact without requiring explicit topology. Second, we propose luminance-structure decoupling to optimize geometric flow in luminance space while maintaining chromatic consistency, preventing color bleeding during structural deformation. Third, we pioneer VLM-as-a-Judge evaluation using multiple large multimodal model assessment (ChatGPT, Claude, Gemini, etc.) to assess artistic authenticity beyond pixel-level metrics.

## II. BACKGROUND

## A. Related Works

Neural Style Transfer. Gatys et al. [16] pioneered neural style transfer using Gram matrix matching of CNN features, treating style as second-order color statistics. Extensions to 3D include NeRF-based methods (ARF [18], StyleRF [19]). With the advent of 3D Gaussian Splatting, StylizedGS [20] pioneered optimization-based transfer for explicit point clouds. Subsequent works focused on efficiency (StyleGaussian [21]) or feature disentanglement (GT 2-GS [22]). However, despite these advances, current 3DGS approaches (including ABC-GS [14]) predominantly treat style as a texture overlay. Even methods that separate geometry from appearance (like GT 2- GS) preserve the underlying photographic topology rather than actively warping it. This âre-texturingâ paradigm fails to capture the structural essence of Post-Impressionist painting. Geometry-Aware Stylization. Several works recognize that artistic style involves geometry. Neural 3D Strokes [23] generates stroke primitives but requires explicit mesh topology. Geometry Transfer [24] aligns depth maps but lacks directional flow guidance. Closest to our work, Kotovenko et al. [9] demonstrated that parameterized brushstrokes outperform pixel optimization in 2D. We extend this insight to 3D: orientation is the syntax of style, and 3DGS primitives must be âcombedâ into directional arrangements to achieve authentic Post- Impressionist aesthetics.

<!-- image-->  
Fig. 3: The Analogy between Artistic Cognition and Computational Simulation. Top: Artistic processâperceiving 3D reality, deciding stroke orientations, creating 2D expression. Bottom: Our computational pipeline mirrors thisâextracting flow from style, modeling as Gaussians, rendering via geometric advection.

<!-- image-->  
Fig. 4: The Projection-Induced Advection Process.

## B. Artistic Motivation: Painting as Dimensional Translation

The Painterâs Cognitive Process: Master painters perform a dimensional translation: perceiving 3D scenes and encoding volumetric form onto 2D surfaces. Van Gogh achieved this not through photographic detail but through directional syntaxâ aligning brushstrokes with principal curvature to construct perceived geometry (Fig. 2). His turbulent, high-anisotropy strokes encode local curvature; Munchâs laminar waves encode psychological resonance [4]. Both share a geometric principle: orientation, not color, encodes 3D structure.

<!-- image-->  
Fig. 5: Overview of the Thinking Like Van Gogh Framework.

We model this artistic cognition as a three-stage process (Fig. 3):

1) 3D Perception: 3DGS captures scene geometry via Gaussian positions and covariancesâthe âpainterâs eyeâ.

2) Orientation Decision: Extract 2D directional flow from paintings, representing the artistâs stroke choices.

3) Geometric Advection: Back-propagate flow gradients to rectify 3D Gaussians, âcombingâ the point cloud into coherent brushworkâthe âpainterâs handâ.

The Translation Problem. Unlike mesh-based NPR [25], 3DGS lacks explicit surface topologyâcomputing continuous tangent fields is ill-posed. We solve this via projection-induced advection (Fig. 4): using 2D flow as a proxy to infer optimal 3D Gaussian arrangements, analogous to how painters validate 3D perception through 2D rendering.

## III. METHODOLOGY

## A. Framework Overview

Our implementation builds upon ABC-GS [14], but diverges in its optimization objective. Rather than prioritizing semantic fidelity and photorealistic consistency, we focus on geometric stylization and painterly abstraction, aiming to reconstruct the physical logic of brushwork.

We formulate stylization as a projection-driven optimization process, in which the agent perceives the 3D scene Sârepresented by a set of unstructured 3D Gaussians G = $\{ ( \mu _ { i } , q _ { i } , s _ { i } , c _ { i } , \alpha _ { i } ) \}$ solely through its 2D rendered projections. The optimization jointly minimizes three types of energies: (i) a flow-alignment energy that enforces coherent, anisotropic brushstroke geometry, (ii) a geometric regularization energy that preserves global 3D structure while allowing controlled deformation, and (iii) an appearance decoupling energy that separates structural stylization from chromatic consistency.

As illustrated in Fig. 5, we realize this objective through a dual-branch decoupled optimization strategy:

â¢ Geometric Advection Branch: Acting as the painterâs âhandâ, this branch utilizes projection analysis to backpropagate 2D flow gradients to the 3D space. It explicitly updates the positions (Âµ) and rotations (q) of the primitives, rectifying the isotropic point cloud into anisotropic constitutive brushstrokes.

â¢ Luminance-Structure Branch: Acting as the painterâs âpaletteâ, this branch employs a luminance-only constraint to optimize the color (c). This ensures that the geometric deformations do not introduce chromatic artifacts, preserving the purity of the styleâs palette.

This decoupled design enables us to unlock the geometric plasticity of 3DGS necessary for impasto effects, without compromising the semantic coherence of the scene.

## B. Flow-Guided Geometric Advection

Our framework reconstructs the volumetric brushstrokes of master artists (e.g., Van Gogh) through a Geometry-First, Color-Second strategy. Since 3D Gaussian Splatting (3DGS) lacks continuous surface topology (mesh), we cannot rely on pre-computed curvature fields. Instead, we propose a meshfree approach that rectifies the 3D primitives directly from 2D artistic cues.

1) Flow-Aware Primitive Rectification: Standard 3DGS initializes Gaussians as isotropic primitives, which fails to capture the strong directionality of impasto-style brushwork.

We therefore introduce a flow-alignment energy that rectifies Gaussian orientation directly from 2D artistic cues.

We extract a dominant local stroke orientation v2D from the style reference using structure tensor analysis, where the leading eigenvector of a local gradient covariance matrix is used as a computationally efficient directional proxy in projection space. For each primitive $\mathcal { G } _ { i }$ , we define an alignment energy that encourages the projected major axis of the Gaussian to follow the artistic flow:

$$
\mathcal { L } _ { \mathrm { a l i g n } } ^ { ( i ) } = 1 - \left| \langle \Pi ( { \bf R } ( { q } _ { i } ) { \bf e } _ { 1 } ) , { \bf v } _ { \mathrm { 2 D } } \rangle \right|\tag{1}
$$

where $\mathbf { e } _ { 1 }$ is the canonical major axis of the Gaussian and Î  denotes perspective projection. This formulation is invariant to sign ambiguity and measures directional consistency in the image plane. Gaussian rotations $q _ { i }$ are updated by backpropagating the gradient of $\mathcal { L } _ { \mathrm { a l i g n } }$ through the differentiable renderer.

2) Gradient-Driven Advection Optimization: With Gaussian orientations rectified to align with image-space stroke directions, we further advect the primitives to form coherent volumetric brushstrokes. Specifically, we unlock the positional parameters $\mu _ { i } ~ \in ~ \mathbb { R } ^ { 3 }$ and update them via gradients of the stylization objective.

Although the stylization losses are defined in the image plane, their influence on $\mu _ { i }$ is realized through the differentiable projection operator $\Pi ( \cdot )$ that maps 3D Gaussian means to screen-space coordinates. Let $\mathbf { u } _ { i } ~ = ~ \Pi ( \mu _ { i } )$ denote the projected 2D mean. By the chain rule, the gradient with respect to the 3D position is given by

$$
\nabla _ { \mu _ { i } } \mathcal { L } = \left( \frac { \partial \Pi ( \mu _ { i } ) } { \partial \mu _ { i } } \right) ^ { \top } \nabla _ { \mathbf { u } _ { i } } \mathcal { L }\tag{2}
$$

where $\nabla _ { \mathbf { u } _ { i } } \mathcal { L }$ is the image-space gradient accumulated by the differentiable renderer. This Jacobian-mediated backpropagation lifts 2D stylization cues into consistent 3D updates under perspective projection.

Because all views share the same set of 3D parameters $\{ \mu _ { i } \}$ , gradients computed from different viewpoints are accumulated on the same primitives during optimization. As a result, the induced advection operates in 3D space rather than as a view-dependent texture deformation, yielding geometrically coherent updates across views when multiple viewpoints are considered.

Direct application of these gradients, however, may introduce spurious motion along the footprint normal of a primitive, causing it to drift off the underlying surface. To constrain the optimization to physically plausible advection, we apply a soft tangential constraint to the optimizer update $\Delta \mu _ { i } .$ Let $\mathbf { n } _ { i } ~ = ~ \mathbf { R } ( q _ { i } ) \mathbf { e } _ { 3 }$ denote the minor-axis direction of the anisotropic Gaussian, corresponding to the local normal of its footprint. We project the update onto the associated tangent plane as

$$
\Delta \mu _ { i } ^ { \mathrm { t a n } } = \Delta \mu _ { i } - \lambda ( \Delta \mu _ { i } \cdot \mathbf { n } _ { i } ) \mathbf { n } _ { i } ,\tag{3}
$$

where Î» controls the strength of the constraint. This operation suppresses normal-direction drift while preserving motion along tangential directions, encouraging primitives to slide coherently along an implicit surface.

In addition, we augment the stylization objective with auxiliary anisotropy terms defined in the image plane. These terms penalize footprint energy orthogonal to the stroke direction while promoting elongation along the stroke tangent. Through the same Jacobian chain, these image-space anisotropy cues further bias the 3D advection toward thin, directionally extended volumetric elements that behave as constitutive brushstrokes across views.

## C. Luminance-Structure Decoupling for Stable Advection

A major challenge in geometric advection is the conflict between structural deformation and appearance preservation. When primitives undergo displacement to align with brushstroke flow, pixel-level correspondence is disrupted. For highly expressive styles like The Starry Night, VGG networks often misinterpret geometric shifts as texture errors, hallucinating chromatic noise that produces a âmuddyâ appearance.

We address this through luminance-structure decoupling (adapted from ARF-Plus [26]). We hypothesize that brushstroke geometry is primarily encoded in luminance, while chromatic information should remain stable. We transform images into YIQ space and restrict the style loss to the luminance (Y) channel:

$$
\mathcal { L } _ { \mathrm { s t y l e } } = \sum _ { \ell } \left\| \phi _ { \ell } ( \mathcal { T } _ { \mathrm { r e n d e r } } ^ { \mathrm { Y } } ) - \phi _ { \ell } ( \mathcal { T } _ { \mathrm { s t y l e } } ^ { \mathrm { Y } } ) \right\| _ { 2 } ^ { 2 }\tag{4}
$$

where $\phi _ { \ell } ( \cdot )$ denotes VGG features at layer â.

To stabilize color appearance, we constrain chromatic statistics in Lab space, matching both means $\pmb { \mu } _ { a b } = ( \mu _ { a } , \mu _ { b } )$ and standard deviations $\pmb { \sigma } _ { a b } = ( \sigma _ { a } , \sigma _ { b } )$ :

$$
\mathcal { L } _ { a b } = \frac { 1 } { 2 } \left( \Vert \pmb { \mu } _ { a b } ^ { \mathrm { r e n d e r } } - \pmb { \mu } _ { a b } ^ { \mathrm { r e f } } \Vert _ { 1 } + \Vert \pmb { \sigma } _ { a b } ^ { \mathrm { r e n d e r } } - \pmb { \sigma } _ { a b } ^ { \mathrm { r e f } } \Vert _ { 1 } \right)\tag{5}
$$

This decoupling liberates geometric advection from color penalties, enabling coherent brushstroke formation without chromatic artifacts.

## IV. EXPERIMENTS

## A. Experimental Setup

We evaluate our method on real-world scenes (LLFF, Tanks & Temples) stylized with Post-Impressionist masterpieces (e.g., The Starry Night). We benchmark against the state-ofthe-art ABC-GS [14]. To ensure fair comparison, we adopt the baselineâs core configuration, including the VGG-16 backbone [27] and loss weighting scheme. Crucially, unlike ABC-GS which disables density control, we enable adaptive densification (clone and split) to populate geometric voids created by advection. Experiments were conducted on a single NVIDIA A100 GPU for 3,000 iterations (â¼5 mins). Detailed settings are provided in the Supplementary Material.

Style Reference  
ABC-GS  
<!-- image-->  
Fig. 6: Qualitative comparisons with the baseline methods (ABC-GS) using references from Van Gogh and Edvard Munch. By prioritizing aesthetic energy over physical accuracy, our method captures the creative intent (the âmindâ) of the original masterpiece.

## B. Qualitative Evaluation

We conduct a visual comparison in Fig. 6. As shown in the zooms, baseline method ABC-GS effectively transfer color statistics but fail to reconstruct the physical stroke geometry. They treat the object as a smooth surface, resulting in a âtexture mappingâ look where brushstrokes are flatly projected, often cutting across structural edges (e.g., the straight textures on the curved truck wheel).

In contrast, our method produces constitutive brushstrokes. Driven by the projection-induced flow, the Gaussian primitives physically rotate and align with the sceneâs curvature. This results in a coherent painterly flow that mimics the artistâs hand, creating a strong sense of relief and directional energy absent in prior works.

## C. Quantitative Evaluation

Quantifying artistic stylization is challenging, as standard metrics often penalize geometric abstraction. We propose a comprehensive protocol combining VLM-based semantic assessment and human user studies.

1) The Misalignment of Standard Metrics: As reported in the Supplementary Material, ArtFID scores exhibit high variance contingent on specific style-scene pairs, resulting in comparable averages despite significant perceptual differences. We attribute this instability to the metricâs sensitivity to lowlevel texture statistics rather than geometric coherence. This lack of discriminative consistency necessitates the semanticaware evaluation introduced below.

TABLE I: VLM-as-a-Judge Evaluation Result
<table><tr><td>Criterion</td><td>Win Rate</td><td>Baseline</td><td>Ours</td></tr><tr><td>Flow Alignment</td><td>85.00%</td><td>7.13</td><td>8.38</td></tr><tr><td>Materiality</td><td>85.83%</td><td>7.13</td><td>8.20</td></tr><tr><td>Aesthetics</td><td>85.83%</td><td>7.31</td><td>8.50</td></tr><tr><td>Average</td><td>85.83%</td><td>7.19</td><td>8.36</td></tr></table>

2) VLM-as-a-Judge: The AI Critic Panel: To overcome the limitations of pixel-based metrics, we established a âPanel of AI Criticsâ comprising GPT-5.1, GPT-4o, Claude 4.5, Claude 3.5, Gork and Qwen 3. We tasked these models to perform randomized pairwise comparisons using a strict âTie-Breakerâ protocol that explicitly penalizes âflat sticker artifactsâ (texture mapping) while rewarding âgeometric flow.â We also recorded an Authenticity Score (1-10) to quantify the gap between digital and physical aesthetics. As detailed in Table I, the panel reached a strong consensus. Our method achieved an average Win Rate of > 87% across geometric metrics.

3) User Study: To evaluate the perceptual quality of our stylization, we conducted a user study with 30 participants, comprising 18 art experts (professionals in design or fine arts) and 12 general laypeople. We randomly selected 4 diverse groups of scenes for blind pairwise comparisons against baseline methods. For each pair, participants were asked to indicate their preference based on three distinct dimensions: Flow Alignment (geometric logic of brushstrokes), Painterly Materiality (impasto feel and texture), and overall Aesthetic

<!-- image-->

<!-- image-->

Fig. 7: The Result of User Study  
<!-- image-->  
Fig. 8: The Results of Ablation Study

Preference.The aggregated results are presented in Fig. 7. Compared to baselines, respondents overwhelmingly preferred our method, particularly in geometric metrics, yielding an average win rate of 82% for Structural Flow and 78% for Materiality across all scenes. Please refer to the Supplementary Material for detailed methodologies and statistical breakdowns.

## D. Ablation Study

To validate our framework, we systematically disable key components (Fig. 8). First, disabling Geometric Advection results in flat texture projections lacking volumetric relief, confirming flow guidance is essential for impasto effects. Second, replacing Color Preservation with standard RGB loss couples geometry and appearance, causing âmuddyâ chromatic artifacts. Finally, enforcing strict regularization (w/o Adaptive Densification) restricts flow magnitude and leads to geometric tearing, whereas our relaxed strategy ensures continuous largescale deformation. Detailed numerical analysis is provided in the Supplement.

## V. CONCLUSION

We presented âThinking Like Van Goghâ, a flow-guided geometric advection framework for 3DGS that prioritizes directional syntax over texture projection. By actively warping geometry to follow artistic flow, we achieve the structural abstraction characteristic of Post-Impressionismâdeliberately sacrificing photographic fidelity for expressive coherence. While we focused on Van Gogh due to his seminal role in defining geometric flow, his influence permeates modern art. Future work will extend this âgeometry-firstâ paradigm to broader artistic movements (e.g., Expressionism, Futurism), further demonstrating that teaching 3D primitives to follow the syntax of orientation is key to the computational replication of artistic cognition. Additional experimental results, theoretical analysis, and implementation details are provided in the supplementary material.

## REFERENCES

[1] V. van Gogh, âLetter to Theo van Gogh, Arles, 26 May 1888 (Letter 490),â Vincent van Gogh: The Letters, WebExhibits / Van Gogh Museum, 1888, in The Letters of Vincent van Gogh, Van Gogh Museum, Amsterdam. [Online]. Available: https://www.webexhibits. org/vangogh/letter/18/490.htm

[2] D. Silverman, Van Gogh and Gauguin: The Search for Sacred Art. Yale University Press, 2000.

[3] D. Bomford, J. Kirby, J. Leighton, and A. Roy, Art in the Making: Van Gogh. London: National Gallery Publications, 1990.

[4] J. M. Aragon, G. Mart Â´ Â´Ä±nez-Mekler, and G. Naumis, âTurbulent Luminance in Impassioned van Gogh Paintings,â Journal of Mathematical Imaging and Vision, vol. 30, pp. 275â283, 2006.

[5] E. Avdeeva and V. Degtyarenko, âArchitectural Space in the Paintings by Vincent van Gogh,â Journal of Siberian Federal University, vol. 13, pp. 838â859, 2020.

[6] R. Arnheim, Art and Visual Perception: A Psychology of the Creative Eye. Berkeley, CA: University of California Press, 1974.

[7] J. Yu, âThe Transcendence of Traditional Concepts in Modern Chinese and Western Painting,â Literature Language and Cultural Studies, 2025.

[8] J. Li, L. Wang, Y. Zhao, and Z. Chen, âRhythmic Brushstrokes Distinguish van Gogh from His Contemporaries: Findings via Automated Brushstroke Extraction,â Pattern Recognition Letters, vol. 146, pp. 40â 47, 2021, uses computer vision to show Van Goghâs rhythmic and directional brushwork uniquely encodes structural perception.

[9] D. Kotovenko, M. Wright, T. Berg-Kirkpatrick, and B. Ommer, âRethinking Style Transfer: From Pixels to Parameterized Brushstrokes,â in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

[10] R. Liu and T. Chan, âGeometric Tight Frame Based Stylometry for Art Authentication of Van Gogh Paintings,â Applied and Computational Harmonic Analysis, 2014, identifies Van Goghâs unique directional brushstroke features using geometric tight frame statistics.

[11] M. Bigerelle and A. Guibert, âFractal and Statistical Characterization of Brushstroke on Paintings,â Surface Topography: Metrology and Properties, vol. 11, 2023, analyzes multiscale topographic signatures of brushstroke orientation and texture.

[12] A. Brachmann and C. Redies, âComputational and Experimental Approaches to Visual Aesthetics,â Frontiers in Computational Neuroscience, vol. 11, p. 102, 2017, links perceptual and computational models of aesthetic structure.

[13] R. Rajbhandari, âRhythm in Painting,â Journal of Fine Arts Campus, 2024, discusses rhythm and orientation as the syntax of pictorial style.

[14] W. Liu, Z. Liu, X. Yang, M. Sha, and Y. Li, âABC-GS: Alignment-Based Controllable Style Transfer for 3D Gaussian Splatting,â in 2025 IEEE International Conference on Multimedia and Expo (ICME), 2025, pp. 1â6.

[15] Q. Fu and J. Yu, âHigh Relief from Brush Painting,â IEEE Transactions on Visualization and Computer Graphics, vol. 25, pp. 2763â2776, 2019, generates 2.5D high-relief from 2D brushstrokes, extending painting structure to 3D form.

[16] L. A. Gatys, A. S. Ecker, and M. Bethge, âImage Style Transfer Using Convolutional Neural Networks,â IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2414â2423, 2016, introduced Gram matrix-based style representation, equating style with color-texture statistics.

[17] Y. Sun and L. Yang, âFrom Pigments to Pixels: A Comparison of Human and AI Painting,â Applied Sciences, 2022, shows that AI-generated paintings lack spatial and emotional coherence present in human works.

[18] K. Zhang, N. Kolkin, S. Bi, F. Luan, Z. Xu, E. Shechtman, and N. Snavely, âArtistic Radiance Fields,â in European Conference on Computer Vision (ECCV), 2022, pp. 717â733.

[19] K. Liu, F. Zhan, Y. Chen, J. Zhang, Y. Yu, A. El Saddik, S. Lu, and E. P. Xing, âStylerf: Zero-shot 3d style transfer of neural radiance fields,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 8338â8348.

[20] D. Zhang, Y.-J. Yuan, Z. Chen, F.-L. Zhang, Z. He, S. Shan, and L. Gao, âStylizedgs: Controllable stylization for 3d gaussian splatting,â IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[21] K. Liu, F. Zhan, M. Xu, C. Theobalt, L. Shao, and S. Lu, âStylegaussian: Instant 3d style transfer with gaussian splatting,â in SIGGRAPH Asia 2024 Technical Communications, 2024, pp. 1â4.

[22] W. Liu, Z. Liu, J. Shu, C. Wang, and Y. Li, âGeometry-aware Texture Transfer for Gaussian Splatting,â arXiv preprint arXiv:2505.15208, 2025.

[23] Z. Tang, Y. Luo, and N. Snavely, âNeural 3D Strokes: Creating Stylized 3D Scenes with Vectorized 3D Brushstrokes,â in ACM SIGGRAPH, 2024.

[24] Y. Wang, X. Liu, Q. Zhang, and J. Yu, âGeometry Transfer for Stylizing Radiance Fields,â IEEE Transactions on Visualization and Computer Graphics, 2023.

[25] A. Hertzmann, âPainterly rendering with curved brush strokes of multiple sizes,â in Proceedings of the 25th annual conference on Computer graphics and interactive techniques, 1998, pp. 453â460.

[26] W. Li, T. Wu, F. Zhong, and C. Oztireli, âArf-plus: Controlling perceptual factors in artistic radiance fields for 3d scene stylization,â in 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2025, pp. 2301â2310.

[27] K. Simonyan and A. Zisserman, âVery deep convolutional networks for large-scale image recognition,â arXiv preprint arXiv:1409.1556, 2014.