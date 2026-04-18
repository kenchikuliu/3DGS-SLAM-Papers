# TranSplat: Surface Embedding-guided 3D Gaussian Splatting for Transparent Object Manipulation

Jeongyun Kim1, Jeongho Noh1, Dong-Guw Lee1 and Ayoung Kim1\*

AbstractâTransparent object manipulation remains a significant challenge in robotics due to the difficulty of acquiring accurate and dense depth measurements. Conventional depth sensors often fail with transparent objects, resulting in incomplete or erroneous depth data. Existing depth completion methods struggle with interframe consistency and incorrectly model transparent objects as Lambertian surfaces, leading to poor depth reconstruction. To address these challenges, we propose TranSplat, a surface embedding-guided 3D Gaussian Splatting method tailored for transparent objects. TranSplat uses a latent diffusion model to generate surface embeddings that provide consistent and continuous representations, making it robust to changes in viewpoint and lighting. By integrating these surface embeddings with input RGB images, TranSplat effectively captures the complexities of transparent surfaces, enhancing the splatting of 3D Gaussians and improving depth completion. Evaluations on synthetic and real-world transparent object benchmarks, as well as robot grasping tasks, show that TranSplat achieves accurate and dense depth completion, demonstrating its effectiveness in practical applications. We open-source synthetic dataset and model: https : / /github. com/jeongyun0609/TranSplat

## I. INTRODUCTION

Manipulating transparent objects is a significant challenge in robotics, as standard depth sensors and depth completion methods often fail to provide accurate reconstructions due to the reflections and refractions inherent in transparent materials. These optical phenomena result in incomplete depth maps, noise, and artifacts, leading to incorrect 3D perception and errors in estimating grasping points.

To address these challenges, previous solutions have focused on hardware or learning-based approaches. Hardwarebased methods use additional sensors, such as thermal infrared cameras [1] or polarized cameras [2, 3], to provide auxiliary depth information. However, thermal cameras are costly to operate, and polarized cameras require specific polarized cues, complicating hardware setups.

More recent solutions emphasize learning-based methods for depth completion using single-view [4, 5] and multiview RGB images [68], facilitated by datasets specifically targeting transparent objects [911]. Multi-view RGB methods, particularly those leveraging Neural Radiance Fields (NeRF), offer more robust depth completion by improving occlusion handling and scale consistency. However, existing methods face three key limitations. First, transparent objects, as non-Lambertian surfaces [12], are highly sensitive to changes in illumination and viewpoint, causing photometric inconsistencies. When using NeRF or 3D Gaussian Splatting (3D-GS) for depth rendering, these inconsistencies introduce noise and artifacts in the depth maps. Second, directly rendering transparent surfaces based solely on RGB images often causes opacity values to collapse to zero [13], resulting in holes in the reconstructed depth. Third, NeRF-based techniques for novel view synthesis of transparent objects, despite recent advancements [7, 8], still suffer from slow inference times.

<!-- image-->  
Fig. 1: TranSplat optimizes 3D Gaussian splatting by jointly training with RGB and surface embeddings as inputs. This approach prevents the opacity of transparent objects from collapsing to zero and ensures smooth rendering, leading to accurate depth completion and reliable grasping points.

In our work, we propose TranSplat (Fig. 1), a novel method combining the strengths of 3D-GS [14] and latent diffusion models to improve the reconstruction of transparent objects. TranSplat uses latent diffusion models to extract surface embeddingsâcontinuous surface representations [13, 15]âfrom transparent object features, ensuring consistent representations that are robust to changes in illumination and viewpoint. This reduces noise and artifacts in depth maps. Additionally, TranSplat introduces a jointly-optimized 3D-GS approach that synthesizes novel views of transparent objects by using both surface embeddings and RGB images. The surface embeddings, acting as surrogate features for non-Lambertian surfaces, prevent the collapse of opacity values and yield accurate depth representation of transparent surfaces. Moreover, employing 3D-GS instead of NeRF not only speeds up rendering but also enhances depth completion accuracy for transparent objects.

<!-- image-->  
Gaussian optimization. Finally, depth is rendered to enable accurate robotic grasping.

TranSplat demonstrates significant improvements in depth completion accuracy on both synthetic datasets and the real-world TRansPose dataset [16]. We further evaluate its effectiveness in depth estimation by applying it to transparent object manipulation, achieving accurate detection of grasping points. The key contributions of our work include:

â¢ Diffusion-based Surface Embeddings: We introduce a novel latent diffusion model specifically designed for transparent objects. This model generates backgroundagnostic surface embeddings that provide consistent representations of transparent surfaces, regardless of viewpoint and illumination changes. By leveraging surface embeddings, our approach achieves enhanced interframe consistency across consecutive RGB images, improving the overall quality of depth completion.

â¢ Gaussian Splatting for Transparent object: We propose an enhanced 3D-GS method through joint optimization of Gaussian kernels using both RGB images and surface embeddings. This approach effectively captures the surface characteristics of transparent objects, achieving accurate depth reconstruction. We further demonstrate the efficacy of our method through real world grasping of transparent objects.

â¢Open-sourcing Synthetic Dataset: Our model and the synthetic datasets used for this work will be opensourced for future development to this field.

## II. RELATED WORK

## A. Explicit Representation for Robot Manipulation

In robot manipulation, explicit object representations such as keypoints [17] and object poses have been commonly used, but recent studies suggest that continuous surface representations, like SurfEmb [15], offer better modeling capabilities, especially for symmetric objects [18]. SurfEmb facilitates 2D-3D matching by generating dense features from 3D CAD models; however, its reliance on CAD models and the need for separate networks for each object limit its scalability. To address these issues, NeuSurfEmb [19] employs NeRF to create large-scale synthetic datasets, enabling dense correspondence matching without CAD models. In our work, we leverage SurfEmb for transparent objects due to its scene-agnostic nature, which ensures consistent representation across consecutive frames, making it effective for dynamic environments.

## B. Latent Diffusion for Representation Generation

With the growing popularity of latent diffusion models for image generation, these models have also demonstrated versatility in various vision tasks, such as depth estimation [20], object detection [21], optical flow [22], and visual navigation [23]. In robotic manipulation, diffusion models have been utilized to formulate representations for pose estimation. A notable example is 6D-Diff [24], which leverages diffusion models to generate keypoint representations, resulting in improved pose estimation accuracy. To the best of our knowledge, our work is the first to employ latent diffusion models to generate explicit representations of transparent objects in the form of SurfEmb.

## C. Depth Completion for Grasping Transparent Objects

Depth completion for transparent objects presents unique challenges that are still being addressed by the research community. Supervised methods rely on paired image-depth data from existing datasets [7, 11, 16], but obtaining accurate 3D CAD models for novel objects is difficult. Moreover, achieving visual fidelity in synthetic data and obtaining precise ground truth in real data remain challenging, leading to reduced performance in out-of-domain scenarios and limiting effectiveness in practical applications like robotic grasping.

Recent approaches have used radiance field-based methods [68] for depth completion through 3D scene reconstruction. Although NeRF-based methods, including those using Spherical Harmonics (SH) coefficients, have shown promise in handling non-Lambertian surfaces, they struggle with transparent objects due to inconsistencies caused by reflection and refraction. Concurrent methods have tried to mitigate inter-frame inconsistencies by extracting geometry using object masks [25, 26]. While these techniques achieve higher surface density through MLP outputs, they often face challenges in maintaining consistency and rely heavily on mask priors, which complicates handling the complexities of transparent objects.

## III. METHODS

As shown in Fig. 2, TranSplat operates in two stages. In the first stage, a latent diffusion model is used to extract surface embeddings from each transparent object in the RGB image, providing a consistent representation of the object across different viewpoints. In the second stage, these surface embeddings, combined with the RGB image, are utilized to render depth and reconstruct 3D scenes through 3D-GS.

## A. Diffusion-based Surface Embedding Extraction

To enhance depth completion for transparent objects, TranSplat generates surface embeddings using a latent diffusion model. Inspired by SurfEmb [15], which effectively captures surface characteristics of various objects, we hypothesize that surface embeddings can provide improved depth completion and a viewpoint-agnostic representation for transparent objects.

To train TranSplat, four data components are required: input RGB image, corresponding mask for transparent object, text condition, and ground truth surface embedding. We trained the model in SurfEmb [15] to generate surface embedding ground truth. However, SurfEmb relies on objectspecific networks trained using 3D CAD models, limiting its scalability to real-world scenarios with unknown objects. In our work, we adopt a more generalizable approach where we leverage a category-level training approach, enabling the network to generate similar features for objects within the same category rather than assigning an object specific CAD model. This allows the model to generalize to a wider range of unseen objects, making it more practical for realworld applications. The modified SurfEmb network is used to generate ground truths for training.

To extract surface embeddings from RGB images with a latent diffusion model, we concatenate latents generated from the image mask and the mask-multiplied RGB image in the forward process. Text conditioning, consisting of categorical descriptions of objects, is also applied (See Fig. 2 green box). Using ControlNet [27] architecture, we employ the cropped RGB image as input control. This control helps guide surface embedding generation for specific objects, particularly in scenes with multiple clustered objects. Examples of the generated surface embeddings are shown in Fig. 3

<!-- image-->

<!-- image-->  
(a) Synthetic unseen object  
(b) Real world unseen object  
Fig. 3: Surface embeddings visualization for unseen transparent objects.

## B. Gaussian Splatting for Transparent Objects

1) Color and Depth Rendering for 3D Gaussian Splatting: To achieve faster rendering speeds than existing NeRF models, we use 3D-GS for depth completion of transparent objects. 3D-GS represents 3D scenes as a collection of Gaussian distributions, with each Gaussian kernel parameterized by its position, color, size, orientation, and visibility. This approach enables smooth and realistic scene rendering. The color and depth of the rendered scenes are computed using these Gaussian attributes, as shown in (1) and (2).

$$
\begin{array} { c } { C = \displaystyle \sum _ { j \in N } c _ { j } \cdot \alpha _ { j } \cdot T _ { j } \mathrm { , ~ w h e r e ~ } T _ { j } = \displaystyle \prod _ { k = 1 } ^ { j - 1 } ( 1 - \alpha _ { k } ) } \\ { D = \frac { \sum _ { j \in N } d _ { j } \cdot \alpha _ { j } \cdot T _ { j } } { \sum _ { j \in N } \alpha _ { j } \cdot T _ { j } } } \end{array}\tag{1}
$$

(2)

where $c , d , \alpha , T$ each represents kernel color, kernel depth, opacity, and the accumulated transmittance for the $j _ { t h }$ observed Gaussian kernel [2830].

2) Joint Gaussian Optimization for Transparent Objects: Applying 3D-GS to non-Lambertian surfaces, such as transparent objects, often results in low opacity values and reduced Î± coefficients. Consequently, the Gaussian kernels on transparent surfaces are obstructed during the splatting process, leading to incomplete depth reconstruction. This issue is further exacerbated by varying backgrounds and viewpoints, reducing depth accuracy.

To address this, TranSplat modifies the 3D-GS rendering process by incorporating surface embedding coefficients. Unlike prior methods that rely solely on rasterizing RGB images, TranSplat rasterizes reconstructed images using the SH coefficients for both RGB and surface embeddings. This dual rasterization allows for independent rendering of both RGB images and surface embeddings. The modified rendering equation is demonstrated in (3) and (4).

$$
C _ { R G B } = \sum _ { j \in N } c _ { R G B , j } \cdot \alpha _ { j } \cdot T _ { j }\tag{3}
$$

$$
C _ { S u r f } = \sum _ { j \in N } c _ { S u r f , j } \cdot \alpha _ { j } \cdot T _ { j }\tag{4}
$$

Moreover, we also reformulate the gaussian optimize loss function to consider images formulated by both RGB and surface embeddings, as shown in (5).

<!-- image-->  
Fig. 4: Depth completion results of TRansPose (Top) and ClearPose (Bottom) synthetic sequences.

TABLEe clet esultnheTRansPosBe esult hlteolSnd bs.
<table><tr><td colspan="4"></td><td colspan="3">Dataset: Synthetic Transpose</td><td colspan="5"></td></tr><tr><td>Evaluation Metric</td><td></td><td></td><td></td><td>MAEâ Residual</td><td>Ours</td><td>Ours</td><td></td><td></td><td>RMSE â</td><td>Residual Ours</td><td>Ours</td></tr><tr><td>Test Sequences</td><td>3D-GS</td><td>SuGar</td><td>Dex-NeRF</td><td>NeRF</td><td>(w/o RGB)</td><td>(w/ RGB)</td><td>3D-GS</td><td>SuGar</td><td>Dex-NeRF</td><td>NeRF (w/o RGB)</td><td>(w/ RGB)</td></tr><tr><td>1</td><td>0.0338</td><td>0.2720</td><td>0.0882</td><td>0.1365</td><td>0.2809</td><td>0.0406</td><td>0.0478</td><td>0.4213</td><td>0.2523 0.4130</td><td>0.4503</td><td>0.0537</td></tr><tr><td>2</td><td>0.1002</td><td>0.0790</td><td>0.0644</td><td>0.0402</td><td>00852</td><td>00272</td><td>0.1796</td><td>0.1601 0.2522</td><td>0.1186</td><td>0.1690</td><td>0 0585</td></tr><tr><td>3</td><td>0.0805</td><td>X</td><td>0.3423</td><td>0.1793</td><td>0.0292</td><td>0.0199</td><td>0.1001</td><td>X</td><td>0.6294 0.4066</td><td>0.0581</td><td>0.0377</td></tr><tr><td>5</td><td>0.1648</td><td>0.0874</td><td>0.2839</td><td>0.0549</td><td>0.0578</td><td>0.0280</td><td>0.2272</td><td>0.1523</td><td>0.8764 0.1907</td><td>0.1358</td><td>0.0648</td></tr><tr><td>6</td><td>0.0867</td><td>0.0637</td><td>0.0641</td><td>0.0450</td><td>0.0633</td><td>0.0266</td><td>0.1618</td><td>0.1296</td><td>0.2548 0.1317</td><td>0.1508</td><td>0.0589</td></tr><tr><td>7</td><td>0.1059</td><td>0.1018</td><td>0.2660</td><td>0.1367</td><td>0.0549</td><td>0.0275</td><td>0.1764</td><td>0.1723</td><td>0.5546 0.2700</td><td>0.1389</td><td>0.0558</td></tr><tr><td>8</td><td>0.2441</td><td>0.1056</td><td>0.6620</td><td>0.3431</td><td>0.0730</td><td>0.0357</td><td>0.3103</td><td>0.1841</td><td>1.4330 06932</td><td>0.1474</td><td>0.0857</td></tr><tr><td>9</td><td>0.2008</td><td>0.1870</td><td>0.1595</td><td>0.3047</td><td>0.1513</td><td>0.1111</td><td>0.2915</td><td>0.3024</td><td>0.3913 0.4908</td><td>0.2483</td><td>0.1743</td></tr><tr><td>10</td><td>0.1848</td><td>0.0758</td><td>0.3870</td><td>0.0965</td><td>00486</td><td>0.0255</td><td>0.2520</td><td>0.1460</td><td>1.0841 0.4234</td><td>0.1140</td><td>0.0559</td></tr><tr><td>11</td><td>0.0504</td><td>0.2128</td><td>0.0419</td><td>0.0434</td><td>0.3103</td><td>0.0452</td><td>0.0762</td><td>0.3708</td><td>0.1237 0.1301</td><td>0.4835</td><td>0.0673</td></tr><tr><td>12</td><td>0.0562</td><td>0.0921</td><td>0.0521</td><td>0.1084</td><td>0.2008</td><td>0.0470</td><td>0.0903</td><td>0.1723</td><td>0.3075 0.3747</td><td>0.3613</td><td>0.0779</td></tr><tr><td>13</td><td>0.0426</td><td>0.1340</td><td>0.0399</td><td>0.1961</td><td>0.2777</td><td>0.0361</td><td>0.0719</td><td>0.3077</td><td>0.1515 0.5154</td><td>0.4938</td><td>0.0661</td></tr><tr><td>14</td><td>0.0521</td><td>0.2654</td><td>0.2815</td><td>0.1768</td><td>0.4307</td><td>0.0461</td><td>0083</td><td>0.4851</td><td>0.9132 0.6196</td><td>0.6325</td><td>0.0809</td></tr><tr><td>15</td><td>0.0702</td><td>0.1097</td><td>0.6331</td><td>0.3125</td><td>0.1553</td><td>0.0606</td><td>01599</td><td>0.1651</td><td>1.5973 0.7698</td><td>0.3031</td><td>0.1250</td></tr></table>

TABLE DeuCro.B u oS .

<table><tr><td>Evaluation Metric</td><td colspan="6">MAEâ</td><td colspan="6">RMSEâ</td></tr><tr><td>Test Sequences</td><td>3D-GS</td><td>SuGar</td><td>Dex-NeRF</td><td>Residual NeRF</td><td>Ours (w/o RGB)</td><td>Ours (w/ RGB)</td><td>3D-GS</td><td>SuGar</td><td>Dex-NeRF</td><td>Residual NeRF</td><td>Ours (w/o RGB)</td><td>Ours (w/ RGB)</td></tr><tr><td>1</td><td>0.2152</td><td>0.1438</td><td>0.6038</td><td>0.5403</td><td>0.1534</td><td>0.0519</td><td>0.2984</td><td>0.2604</td><td>1.4545</td><td>1.2959</td><td>0.3159</td><td>0.1244</td></tr><tr><td>2</td><td>0.1799</td><td>0.1483</td><td>0.2773</td><td>0.1401</td><td>0.1376</td><td>0.0739</td><td>0.2246</td><td>0.2139</td><td>0.4952</td><td>0.4732</td><td>0.2502</td><td>0.1979</td></tr><tr><td>3</td><td>0.1835</td><td>0.0876</td><td>0.4600</td><td>0.2405</td><td>0.1672</td><td>0.0548</td><td>0.3315</td><td>0.1700</td><td>0.9566</td><td>0.7027</td><td>0.3425</td><td>0.1014</td></tr><tr><td>4</td><td>0.1950</td><td>0.1155</td><td>0.0465</td><td>0.0935</td><td>0.1871</td><td>0.0858</td><td>0.3893</td><td>0.2868</td><td>0.1808</td><td>0.2300</td><td>0.3975</td><td>0.1978</td></tr><tr><td>5</td><td>0.1967</td><td>0.1725</td><td>0.0786</td><td>0.0335</td><td>0.1693</td><td>0.0343</td><td>0.3419</td><td>0.3342</td><td>0.3284</td><td>0.1141</td><td>0.3457</td><td>0.0622</td></tr><tr><td>6</td><td>0.3300</td><td>0.3566</td><td>0.0582</td><td>0.0406</td><td>0.3377</td><td>0.0364</td><td>0.5041</td><td>0.5101</td><td>0.2471</td><td>0.1236</td><td>0.5648</td><td>0.0580</td></tr></table>

$$
L = \frac { 1 } { 2 } L _ { R G B } + \frac { 1 } { 2 } L _ { S u r f }\tag{5}
$$

$$
L _ { R G B } = ( 1 - \lambda ) | \hat { I } _ { R G B } - I _ { R G B } | + \lambda \mathrm { D } \mathrm { - } \mathrm { S } \mathrm { S } \mathrm { I } \mathrm { M } ( \hat { I } _ { R G B } , I _ { R G B } )\tag{6}
$$

$$
\boldsymbol { L _ { S u r f } } = ( 1 - \lambda ) | \hat { I } _ { S u r f } - I _ { S u r f } | + \lambda \mathrm { D } \mathrm { - } \mathrm { S S I M } ( \hat { I } _ { S u r f } , I _ { S u r f } )\tag{7}
$$

where $I _ { R G B }$ is the RGB image, $I _ { S u r f }$ is the surface embeddings image, and $\lambda = 0 . 2$ This combined loss function optimizes both RGB content and surface features, providing additional supervision to the surfaces of transparent objects. During backward gradient propagation, the Gaussian kernels' mean, covariance, and Î± values are shared between the RGB and surface embedding images (See Fig. 2 blue box). This joint optimization ensures consistent updates of the SH coefficients for both representations, allowing the surface embeddings to prevent opacity values, Î±, from collapsing to zero on transparent object surfaces.

## IV. EXPERIMENT

## A. Experiment Setup

1) Datasets: We evaluated the performance of TranSplat on completing depth from novel rendered views using three datasets: one real-world transparent object dataset with known categories and two synthetic datasets. The first synthetic dataset contains identical objects to those in the realworld dataset, while the second has unseen object models but within the same categories. All synthetic datasets were rendered using BlenderProc [31].

For the real-world dataset, we used the TransPose benchmark [16], which consists of multispectral, multiview sequential images of transparent objects across 20 categories. Each sequence contains 52 and 53 images with corresponding object depth ground truths for training and testing, respectively. For the synthetic datasets, we created two versions: Synthetic TransPose and Synthetic ClearPose. The Synthetic TransPose dataset was rendered using 3D CAD models provided by the real TransPose dataset, matching both the categories and specific objects. In contrast, the Synthetic ClearPose dataset features different object models within the same categories, designed to test TranSplat's performance on unseen objects. Both synthetic datasets contain 100 sequential images per sequence with ground truth depths for training and testing.

<!-- image-->  
Fig. 5: Depth completion results of TRansPose test sequence 7 and 26.

TABLE  Depth copletion results or Real TRansPose. Best esults highlighted in bold; Second bes ndine.
<table><tr><td></td><td colspan="7">Dataset: Real TRansPose</td><td colspan="3"></td></tr><tr><td>Evaluation Metric</td><td></td><td></td><td>MAEâ</td><td></td><td></td><td></td><td></td><td>RMSE â</td><td></td><td></td></tr><tr><td>Test Sequences</td><td>3D-GS</td><td>SuGar</td><td>Dex-NeRF</td><td>Ours (w/o RGB)</td><td>Ours (w/ RGB)</td><td>3D-GS</td><td>SuGar</td><td>Dex-NeRF</td><td>Ours (w/o RGB)</td><td>Ours (w/ RGB)</td></tr><tr><td>1</td><td>0.0334</td><td>0.0377</td><td>0.0345</td><td>0.0175</td><td>0.0139</td><td>0.0588</td><td>0.0644</td><td>0.0891</td><td>0.0454</td><td>0.0232</td></tr><tr><td>2</td><td>0.0809</td><td>0.0930</td><td>0.0958</td><td>0.0181</td><td>0.0270</td><td>0.1377</td><td>0.1727</td><td>0.2494</td><td>0.0513</td><td>0.0561</td></tr><tr><td>3</td><td>0.0606</td><td>0.0666</td><td>0.0615</td><td>0.0225</td><td>0.0246</td><td>0.0994</td><td>0.1076</td><td>0.2244</td><td>0.0561</td><td>0.0534</td></tr><tr><td>4</td><td>0.0373</td><td>0.0445</td><td>0.0385</td><td>0.0261</td><td>0.0189</td><td>0.0757</td><td>0.0905</td><td>0.1485</td><td>0.0712</td><td>0.0694</td></tr><tr><td>5</td><td>0.0467</td><td>0.0544</td><td>0.0471</td><td>0.0165</td><td>0.0223</td><td>0.0773</td><td>0.0957</td><td>0.1959</td><td>0.0539</td><td>0.0527</td></tr><tr><td>6</td><td>0.1067</td><td>0.1196</td><td>0.0929</td><td>0.0162</td><td>0.0367</td><td>0.1799</td><td>0.2067</td><td>0.2764</td><td>0.0464</td><td>0.1011</td></tr><tr><td>7</td><td>0.0500</td><td>0.0520</td><td>0.0398</td><td>0.0291</td><td>0.0185</td><td>0.0881</td><td>0.0927</td><td>0.1296</td><td>0.0875</td><td>0.0478</td></tr><tr><td>8</td><td>0.0744</td><td>0.0944</td><td>0.0920</td><td>0.0182</td><td>0.0264</td><td>0.1300</td><td>0.1827</td><td>0.2904</td><td>0.0505</td><td>0.0580</td></tr><tr><td>9</td><td>0.0566</td><td>0.0673</td><td>0.1807</td><td>0.0229</td><td>0.0229</td><td>0.0976</td><td>0.1293</td><td>0.5911</td><td>0.0704</td><td>0.0702</td></tr><tr><td>10</td><td>0.0592</td><td>0.0607</td><td>0.0550</td><td>0.0245</td><td>0.0257</td><td>0.0997</td><td>0.1191</td><td>0.2284</td><td>0.0584</td><td>0.0691</td></tr><tr><td>11</td><td>0.0944</td><td>0.0931</td><td>0.0626</td><td>0.0278</td><td>0.0368</td><td>0.1745</td><td>0.1957</td><td>0.1906</td><td>0.0762</td><td>0.1036</td></tr><tr><td>12</td><td>0.0404</td><td>0.0448</td><td>0.0473</td><td>0.0225</td><td>0.0189</td><td>0.0951</td><td>0.1231</td><td>0.1360</td><td>0.0719</td><td>0.0587</td></tr><tr><td>13</td><td>0.0431</td><td>0.0442</td><td>0.0507</td><td>0.0292</td><td>0.0172</td><td>0.0910</td><td>0.0946</td><td>0.1492</td><td>0.0855</td><td>0.0441</td></tr><tr><td>25</td><td>0.0726</td><td>0.0618</td><td>0.1829</td><td>0.0549</td><td>0.0191</td><td>0.1250</td><td>0.1394</td><td>0.4837</td><td>0.1279</td><td>0.0406</td></tr><tr><td>26</td><td>0.0404</td><td>0.0480</td><td>0.1831</td><td>0.0141</td><td>0.0191</td><td>0.0682</td><td>0.1043</td><td>0.3041</td><td>0.0388</td><td>0.0499</td></tr></table>

2) Implementation Details: For TranSplat, we trained both the latent diffusion-based surface embeddings extractor and the 3D-GS for neural rendering. Built on ControlNet [27], we froze the latent diffusion UNet and kept the ControlNet counterpart trainable. Training was performed on 256 Ã 256 images from both real and synthetic TRansPose datasets, with a batch size of 32 and a learning rate of 1.0e-6, using the AdamW optimizer with a cosine scheduler. For the 3D-GS, we followed the settings described in 3D-GS [14]. All models were trained on four Nvidia A6000 GPUs. Further details on the training configurations are available on our project page.

3) Baselines: We used four models as baselines for our evaluations: DexNeRF [7] and Residual-NeRF [8], which are recent models for novel view completion of transparent objects, as well as 3D-GS [14] and SuGaR [32], an object surface-aligned 3D-GS model. For TranSplat, we present two variations: one with RGB image input control (w/ RGB) and one without it (w/o RGB). To assess depth completion performance across the baselines and TranSplat, we used mean average error (MAE) and root mean squared error (RMSE) to compare the absolute depth measurements between the ground truths and the rendered views.

## B. Evaluation on Synthetic Datasets

1) Evaluation on Synthetic TRansPose: As shown in Table. I, TranSplat achieves the best depth completion performance, outperforming all baseline models in terms of both MAE and RMSE across all sequences. Unlike other models that rely directly on raw images of transparent objects as inputs, TranSplat leverages surface embeddings as an alternative representation, leading to a significant improvement in depth completion. Additionally, TranSplat does not require extensive volume density tuning or separate residual background imagesâboth impractical for robotics applications. 3D-GS methods often yield near-zero opacity values due to the non-Lambertian nature of transparent objects. Specifically, in synthetic TRansPose dataset sequence 3, the overall darkness of the dataset causes the opacities of Gaussians for transparent objects to converge to zero, resulting in the failure of SuGaR during the pruning step as no Gaussians remain after pruning for further optimization. In contrast, TranSplat's surface embeddings serve as surrogate features that accurately estimate opacity values on transparent surfaces.

The qualitative results, presented in Fig. 4, further support these findings. Most baseline methods struggle to capture the depth along the edges of transparent objects. Even methods like Dex-NeRF, which do capture some edge details, display incomplete depth with holes around transparent surfaces. This is due to conventional NeRF-based methods neglecting the opacity values for transparent surfaces. In contrast, by incorporating the unique properties of transparent objects and supplementing 3D gaussian optimization with surface embeddings, TranSplat achieves complete and dense reconstructions around transparent object surfaces.

TABLE IV: Depth completion results on reducing the number of images. Best results highlighted in bold; Second best in underlines.
<table><tr><td>Evaluation Metric</td><td colspan="3">MAE â</td><td colspan="3">RMSE â</td></tr><tr><td>Dataset</td><td>Synthetic TRansPose</td><td>Real TRansPose</td><td>Synthetic ClearPose</td><td>Synthetic TRansPose</td><td>Real TRansPose</td><td>Synthetic ClearPose</td></tr><tr><td>3D-GS</td><td>0.1066</td><td>0.0598</td><td>0.2167</td><td>0.1618</td><td>0.1065</td><td>0.3483</td></tr><tr><td>SuGar</td><td>0.1327</td><td>0.0655</td><td>0.1707</td><td>0.2362</td><td>0.1279</td><td>0.2959</td></tr><tr><td>Dex-nerf</td><td>0.2347</td><td>0.0839</td><td>0.2541</td><td>0.6291</td><td>0.2461</td><td>0.6104</td></tr><tr><td>Residual-nerf</td><td>0.1503</td><td>X</td><td>0.1814</td><td>0.3910</td><td>X</td><td>0.4899</td></tr><tr><td>Ours(w/o RGB)</td><td>0.1740</td><td>0.0240</td><td>0.1921</td><td>0.2903</td><td>0.0661</td><td>0.3694</td></tr><tr><td>Ours(w/ RGB)</td><td>0.0406</td><td>0.0232</td><td>0.0562</td><td>0.0757</td><td>0.0599</td><td>0.1236</td></tr><tr><td>Ours(w/ RGB, 1/2)</td><td>0.0449</td><td>0.0360</td><td>0.0563</td><td>0.0820</td><td>0.0875</td><td>0.1185</td></tr><tr><td>Ours(w/ RGB, 1/4)</td><td>0.0418</td><td>0.1120</td><td>0.0646</td><td>0.0766</td><td>0.2154</td><td>0.1161</td></tr></table>

2) Evaluation on Unseen Synthetic: We evaluated TranSplat's robustness to unseen objects using the Synthetic ClearPose dataset, as shown in Table II. Consistent with previous findings, TranSplat achieves the best depth completion performance across all test sequences on average. While it slightly underperforms compared to Residual-NeRF on sequence 5 and Dex-NeRF on sequence 4, the performance gaps are minimal. We attribute the lower performance on sequence 4 to the fact that the objects in this sequence differ significantly from the CAD models used to train the latent diffusion model in the TransPose dataset. Despite this, TranSplat demonstrates the highest robustness to unseen objects, proving its effectiveness across different categories rather than being object-specific. Qualitatively, as shown in Fig. 4, other methods fail to accurately render the depth images of transparent objects, whereas TranSplat consistently succeeds.

## C. Evaluation on Real-world Dataset

As shown in Table. III, TranSplat outperforms baseline models in depth completion when evaluated on the real-world TRansPose dataset. However, unlike the consistent results seen in the synthetic dataset evaluation, the best performance in the real-world dataset alternates between TranSplat models with and without RGB images as conditioning input. In the synthetic dataset, incorporating RGB images as input control to the latent diffusion model consistently leads to better results. This is because RGB images provide valuable context and guidance, allowing the model to better attend to transparent objects during depth completion, as illustrated in Fig. 5. In contrast, in the real-world scenario, using RGB input does not always improve performance. The RGB images in real-world settings are often affected by factors like poor lighting, image sensor noise, and severe occlusion between objects, which can negatively impact the model's effectiveness when using RGB conditioning. Additionally, it is important to note that we were unable to evaluate TranSplat against Residual NeRF for the real-world dataset because Residual NeRF requires background images, which were not provided in the TransPose dataset.

## D. Computational Efficiency Analysis

A practical approach to reducing computation time in robotics applications is to decrease the number of images used for rendering. This is particularly relevant when the sensor's sampling rate does not support high frame rates, resulting in sparse image outputs. To evaluate this, we analyzed the accuracy-efficiency trade-off of TranSplat by varying the number of images used. As shown in Table IV, reducing the number of images leads to a slight decrease in performance. However, despite this minor degradation, TranSplat still outperforms other baseline models that use more images. Although TranSplat's use of diffusion models can result in slower inference times with a full sequence of images, reducing the number of rendered images can significantly enhance computational efficiency with only a minor drop in accuracy. This also simplifies the system overhead by allowing for a lower sampling rate of visual sensors while maintaining competitive performance.

<!-- image-->  
Fig. 6: The first column is RGB images. The second column shows the grasp planning locations from Graspnet.

## E. Extension to Transparent Object Grasping

To explore the feasibility of extending TranSplat for robot manipulation through grasping, we tested its performance using a commercial robot arm. We used the Franka Emika Panda to capture a series of RGB images of unseen transparent objects, as shown in the Fig. 6. These images were then used to generate input point clouds by combining RGB data with the corresponding depth rendered by TranSplat. To determine the grasping points, we employed the pretrained GraspNet model [33], which generates grasp points from input point clouds. The point clouds were created using RGB images and the depth outputs generated by TranSplat. As shown in Fig. 6, GraspNet successfully identifies valid grasping points using the depth rendered from TranSplat. This demonstrates that the depth produced by TranSplat provides accurate depth information that can be effectively used for robotic manipulation of transparent objects. We encourage readers to refer to the supplementary materials for more details.

## V. CONCLUSION

Accurately capturing the depth of transparent objects remains a significant challenge for conventional depth sensors, which often struggle with transparency. Existing methods using radiance field-based techniques attempt to address this by rendering depth from novel views, but they often fail to handle transparent surfaces adequately, leading to incomplete depth renderings. In this work, we introduced TranSplat, which overcomes this limitation by incorporating surface embeddings generated through latent diffusion models. TranSplat consistently outperforms existing methods in accurately capturing the depth of transparent objects in both synthetic and real-world datasets, including practical applications in robot grasping tasks. For future work, we plan to enhance TranSplat by estimating uncertainties in the input RGB images used for conditioning, further improving its robustness and applicability.

[1] D. Huo et al., "Glass segmentation with RGB-thermal image pairs," IEEE Trans. Image Processing, vol. 32, pp. 19111926, 2023.

[2] H. Mei et al., "Glass segmentation using intensity and spectral polarization cues," in Proc. IEEE Conf. on Comput. Vision and Pattern Recog., 2022, pp. 12 62212631.

[3] A. Kalra et al., "Deep polarization cues for transparent object segmentation," in Proc. IEEE Conf. on Comput. Vision and Pattern Recog., 2020, pp. 86028611.

[4] L. Zhu et al., "Rgb-d local implicit function for depth completion of transparent objects," in Proc. IEEE Conf. on Comput. Vision and Pattern Recog., 2021, pp. 46494658.

[5] H. Fang et al., "Transcg: A Large-Scale Real-World Dataset for Transparent Object Depth Completion and a Grasping Baseline," IEEE Robot. and Automat. Lett., vol. 7, no. 3, pp. 73837390, 2022.

[6] J. Kerr et al., "Evo-nerf: Evolving NeRF for sequential robot grasping of transparent objects," in 6th annual conference on robot learning.

[7] J. Ichnowski et al., "Dex-Nerf: Using a Neural Radiance Field to Grasp Transparent Objects," in 6th annual conference on robot learning, 2022, pp. 526536.

[8] B. P. Duisterhof, Y. Mao, S. H. Teng, and J. Ichnowski, "Residual-nerf: Learning residual nerfs for transparent object manipulation," in Proc. IEEE Intl. Conf. on Robot. and Automat., 2024.

[9] P. Wang et al., "Phocal: A multi-modal dataset for categorylevel object pose estimation with photometrically challenging objects," in Proc. IEEE Conf. on Comput. Vision and Pattern Recog., 2022, pp. 21 22221 231.

[10] D. Bashkirova et al., "Zerowaste dataset: Towards deformable object segmentation in cluttered scenes," in Proc. IEEE Conf. on Comput. Vision and Pattern Recog., 2022, pp. 21 147 21 157.

[11] X. Chen et al., "Clearpose: Large-scale transparent object dataset and benchmark," in Proc. European Conf. on Comput. Vision, 2022, pp. 381396.

[12] S. Sajjan, M. Moore, M. Pan, G. Nagaraja, J. Lee, A. Zeng, and S. Song, "Clear grasp: 3d shape estimation of transparent objects for manipulation," in 2020 IEEE international conference on robotics and automation (ICRA). IEEE, 2020, pp. 36343642.

[13] J. Lee, S. M. Kim, Y. Lee, and Y. M. Kim, "Nfl: Normal field learning for 6-dof grasping of transparent objects," IEEE Robot. and Automat. Lett., 2023.

[14] B. Kerbl, G. Kopanas, T. LeimkÃ¼hler, and G. Drettakis, "3d gaussian splatting for real-time radiance field rendering." ACM Trans. Graph., vol. 42, no. 4, pp. 1391, 2023.

[15] R. L. Haugaard and A. G. Buch, "Surfemb: Dense and continuous correspondence distributions for object pose estimation with learnt surface embeddings," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 67496758.

[16] J. Kim, M.-H. Jeon, S. Jung, W. Yang, M. Jung, J. Shin, and A. Kim, "Transpose: Large-scale multispectral dataset for transparent object," The International Journal of Robotics Research, p. 02783649231213117, 2024.

[17] M.-H. Jeon et al., "Ambiguity-Aware Multi-Object Pose Optimization for Visually-Assisted Robot Manipulation ," IEEE Robot. and Automat. Lett., 2022.

[18] R. L. Haugaard and T. M. Iversen, "Multi-view object pose estimation from correspondence distributions and epipolar geometry," in 2023 IEEE International Conference on Robotics

and Automation (ICRA). IEEE, 2023, pp. 17861792.

[19] F. Milano, J. J. Chung, H. Blum, R. Siegwart, and L. Ott, "Neusurfemb: A complete pipeline for dense correspondencebased 6d object pose estimation without cad models," arXiv preprint arXiv:2407.12207, 2024.

[20] B. Ke, A. Obukhov, S. Huang, N. Metzger, R. C. Daudt, and K. Schindler, "Repurposing diffusion-based image generators for monocular depth estimation," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 94929502.

[21] S. Chen, P. Sun, Y. Song, and P. Luo, "Diffusiondet: Diffusion model for object detection," in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 19 83019 843.

[22] S. Saxena, C. Herrmann, J. Hur, A. Kar, M. Norouzi, D. Sun, and D. J. Fleet, "The surprising effectiveness of diffusion models for optical flow and monocular depth estimation," Advances in Neural Information Processing Systems, vol. 36, 2024.

[23] A. Sridhar, D. Shah, C. Glossop, and S. Levine, "Nomad: Goal masked diffusion policies for navigation and exploration," in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 6370.

[24] L. Xu, H. Qu, Y. Cai, and J. Liu, "6d-diff: A keypoint diffusion framework for 6d object pose estimation," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 96769686.

[25] X. Chen, J. Liu, H. Zhao, G. Zhou, and Y.-Q. Zhang, "Nerrf: 3d reconstruction and view synthesis for transparent and specular objects with neural refractive-reflective fields," arXiv preprint arXiv:2309.13039, 2023.

[26] A. Ummadisingu, J. Choi, K. Yamane, S. Masuda, N. Fukaya, and K. Takahashi, "Said-nerf: Segmentation-aided nerf for depth completion of transparent objects," arXiv preprint arXiv:2403.19607, 2024.

[27] L. Zhang, A. Rao, and M. Agrawala, "Adding conditional control to text-to-image diffusion models," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 38363847.

[28] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, "4d gaussian splatting for real-time dynamic scene rendering," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 31020 320.

[29] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, "Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 2033120341.

[30] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, "Gaussian splatting slam," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 03918 048.

[31] B. O. Community, Blender - a 3D modelling and rendering package, Blender Foundation, Stichting Blender Foundation, Amsterdam, 2018. [Online]. Available: http://www.blender.org

[32] A. GuÃ©don and V. Lepetit, "Sugar: Surface-aligned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 5354 5363.

[33] H.-S. Fang, C. Wang, M. Gou, and C. Lu, "Graspnet-1billion: A large-scale benchmark for general object grasping," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 11 44411 453.