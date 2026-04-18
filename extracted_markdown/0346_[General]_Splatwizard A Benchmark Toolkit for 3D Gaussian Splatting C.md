# Splatwizard: A Benchmark Toolkit for 3D Gaussian Splatting Compression

Xiang Liu1,3 Yimin Zhou1 Jinxiang Wang1 Yujun Huang1 Shuzhao Xie1 Shiyu Qin1   
Mingyao Hong3 Jiawei Li4 Yaowei Wang2,3 Zhi Wang1 Shu-Tao Xia1 Bin Chen2,\* 1Tsinghua University 2Harbin Institute of Technology, Shenzhen 3Pengcheng Laboratory 4Huawei Technologies Ltd.

## Abstract

The recent advent of 3D Gaussian Splatting (3DGS) has marked a significant breakthrough in real-time novel view synthesis. However, the rapid proliferation of 3DGS-based algorithms has created a pressing need for standardized and comprehensive evaluation tools, especially for compression task. Existing benchmarks often lack the specific metrics necessary to holistically assess the unique characteristics of different methods, such as rendering speed, rate distortion trade-offs memory efficiency, and geometric accuracy. To address this gap, we introduce Splatwizard, a unified benchmark toolkit designed specifically for benchmarking 3DGS compression models. Splatwizard provides an easy-to-use framework to implement new 3DGS compression model and utilize state-of-the-art techniques proposed by previous work. Besides, an integrated pipeline that automates the calculation of key performance indicators, including image-based quality metrics, chamfer distance of reconstruct mesh, rendering frame rates, and computational resource consumption is included in the framework as well. Code is available at Github.

## 1. Introduction

With the rapid progress of computer vision, 3D reconstruction has emerged as a core research area, underpinning downstream applications in virtual reality and embodied intelligence. Among various 3D representations, 3D Gaussian Splatting (3DGS) [23] achieves state-of-the-art rendering quality and real-time performance, enabling a wide range of practical uses.

However, raw 3DGS scenes contain substantial redundancy. A 3DGS represents a scene with millions of Gaussian ellipsoids distributed in 3D space, whose projected splats jointly capture geometry and appearance. In practice, a single scene often requires up to 107 Gaussians, leading to hundreds of megabytes of storage and limiting the deployability of uncompressed 3DGS in real applications. This has driven the development of numerous 3DGS compression techniques, as summarized in Table 1. These methods introduce heterogeneous designs across preprocessing, rendering pipelines, and post-processing stages, resulting in large differences in code structure, system behavior, and implementation logic. Consequently, they exhibit notable trade-offs in FPS, memory footprint, model size, and reconstruction quality, making it difficult to achieve consistent evaluation or identify universally effective improvements.

To address this issue, several works [2, 45] have attempted to summarize or benchmark existing 3DGS compression methods. However, these surveys largely rely on reported results or cover only a subset of representative techniques, leaving the practical selection of a suitable method still unclearâespecially because downstream tasks prioritize different properties. For instance, streaming systems [14, 37, 50] demand extremely low encoding and decoding latency, while robotics applications [16, 27] require high inference throughput and minimal memory usage for onboard deployment.

Besides, another major limitation of existing evaluations is the absence of geometric assessment. Geometric reconstruction is a central problem in computer graphics, and numerous recent works [6, 15, 20, 21] focus on improving the surface reconstruction quality of 3DGS, drawing substantial attention from the community. However, research on 3DGS compression has so far concentrated almost exclusively on photorealistic metrics such as PSNR and SSIM. Although these metrics capture appearance fidelity, they overlook the fact that 3DGS is fundamentally a three-dimensional representation. As a result, existing evaluations fail to reveal how compression affects the geometric integrity of the underlying 3D structure.

To mitigate the aforementioned issues, we have developed a unified training and evaluation framework, characterized by the following features: 1) Standardization. It defines a unified pipeline encompassing data loading, model training, compression encoding, and rendering evaluation, supporting the rapid development of new methods. 2) Decoupling. We decouple components such as the rasterizer, entropy module, and codec as independent modules. This facilitates the combination of different methods and supports the integration of new modules in the future. 3) More Metrics. Based on the same evaluation pipeline, we have not only included metrics such as PSNR but also extended to Gaussians count, peak memory usage, and geometric reconstruction accuracy, in order to provide a more comprehensive benchmark.

<table><tr><td>Method</td><td>Year</td><td>Std Rasterizer</td><td>QAT</td><td>Entropy Model</td><td>Post Pruning</td><td>Compression</td></tr><tr><td>3DGS [23]</td><td>2023</td><td>â</td><td>-</td><td></td><td></td><td>X</td></tr><tr><td>Trimming the Fat [1]</td><td>2024</td><td>â</td><td></td><td></td><td>Ã &gt;</td><td>X</td></tr><tr><td>PUP-3DGS [18]</td><td>2025</td><td>â</td><td></td><td></td><td>â</td><td>X</td></tr><tr><td>Speedy splat [17]</td><td>2025</td><td>â</td><td></td><td></td><td>X</td><td>X</td></tr><tr><td>LightGaussian [11]</td><td>2024</td><td>â</td><td></td><td></td><td>â</td><td>â</td></tr><tr><td>MesonGS [42]</td><td>2024</td><td>â</td><td>-</td><td></td><td>â</td><td>â</td></tr><tr><td>C3DGS [33]</td><td>2024</td><td>X</td><td>Fake</td><td></td><td>â</td><td></td></tr><tr><td>Compact3DGS [26]</td><td>2024</td><td>X</td><td>RVQ</td><td></td><td>X</td><td>â</td></tr><tr><td>HAC [8]</td><td>2024</td><td>X</td><td>Noise</td><td>Gaus</td><td>X</td><td>â</td></tr><tr><td>CAT-3DGS [48]</td><td>2025</td><td>X</td><td>Noise</td><td>Gaus/Laplace</td><td>X</td><td></td></tr></table>

Table 1. Comparison of different methods.

In summary, our contributions are threefold:

â¢ We propose a unified training and evaluation framework for 3DGS compression, which alleviates reproducibility issues of existing methods and lowers the development barrier for new methods.

â¢ We reproduce existing 3DGS compression methods under the proposed framework, providing the community with a fair, reliable, and comprehensive technical benchmark.

â¢ Based on the framework, we have developed a new compression model, ChimeraGS, which has demonstrated competitive results across several dimensions, highlighting the flexibility and effectiveness of this framework.

## 2. Related Work

The high rendering quality and flexibility of 3DGS have motivated research aimed at improving both reconstruction fidelity and storage efficiency. In this section, we review related works on 3D Gaussian Splatting and the methods proposed to compress and optimize the representation.

## 2.1. Novel View Synthesis

Novel view synthesis is a classic problem in graphics research. Many work on the topic focused on image-based rendering [13, 38] and Neural Radiance Fields [5, 32]. In recent years, 3D Gaussian Splatting (3DGS) [23] has emerged as an efficient and expressive representation for novel view synthesis and 3D reconstruction. By explicitly modeling scenes with anisotropic Gaussians, 3DGS achieves highquality rendering and real-time performance, attracting extensive research attention.

Numerous works have improved the rendering pipeline through adaptive density control [17], anti-aliasing [47] and more realistic shading mechanisms [44]. These advances consolidate 3DGS as a versatile representation bridging explicit geometry and neural rendering. Our framework supports various static Gaussian Splatting pipelines with distinct models and optimization objectives, providing a unified platform for evaluation and analysis.

## 2.2. Compression of 3DGS

Despite its rendering efficiency, 3DGS models are storageintensive, prompting research on compact representations. Pruning-based methods[1, 11, 17, 18, 49] remove redundant or visually insignificant Gaussians by analyzing opacity, gradient importance, or visibility contribution, effectively reducing both model size and rendering cost. Entropy-based compression[8, 33, 48] further reduces storage through learned probabilistic coding of Gaussian parameters or by applying rateâdistortion optimization to balance bit usage and rendering accuracy. Although these studies have substantially advanced 3DGS compression, their implementations are often inconsistent, making crossmethod evaluation difficult. To address this, our work introduces a unified framework that integrates representative pruning, quantization, distillation, and entropy coding techniques under a consistent training and evaluation pipeline, facilitating reproducible and comprehensive benchmarking for future research.

## 2.3. Benchmark and Tools

Unlike fully neural network-based models, the 3DGS framework incorporates numerous components beyond conventional neural network architectures, with the rasterizer being a typical example. Furthermore, the inherently complex control flow of 3DGS has generated a need for substantial boilerplate code.

GSplat [46] serves as an open-source, efficient, and userfriendly library that provides an efficient implementation of 3D Gaussian Splatting. GauStudio [45] serves as a framework for modeling 3D Gaussian Splatting to provide standardized, plug-and-play components for users to easily customize and implement a 3DGS pipeline. GSCodec Studio [28] focuses specifically on Gaussian Splat compression, offering a modular design for integrating and benchmarking different compression strategies such as pruning, quantization, and entropy coding. Beyond tools, 3DGS.zip [3] introduces a comprehensive compression for 3DGS compression task, and compared the characteristics of various methods.

Our work is also inspired by these studies and possesses unique characteristics. The most representative aspect is the design of a flexible modular mechanism, which integrates over 10 different rasterizers and other modules including entropy estimation and encoding. Furthermore, based on a scheduler mechanism, we have implemented a more streamlined template for Gaussian Splatting models. Using this foundation, we have reproduced several representative methods and conducted comprehensive performance comparisons.

## 3. Architecture

In Fig. 1, we conceptualize our Splatwizard framework. The primary objective of this framework is to provide a unified and reusable training and evaluation pipeline, thereby simplifying the implementation and research of future GS models. In the subsequent sections, we will provide a detailed introduction to the key mechanisms and functional modules of Splatwizard.

## 3.1. Dynamic and AOT module loading

A challenge in 3DGS research lies in the frequent need to modify or extend the rasterizer [11, 12, 21, 29]. These modifications often tailored to specific tasks and introduce a host of non-research-related hurdles that hinder productivity. This problem is further compounded in compressionfocused 3DGS tasks, where the integration of diverse entropy encoders [8, 48] adds another layer of complexity. These encoders, often implemented in C++ or CUDA for efficiency, introduce additional build dependencies, versioning conflicts, and compatibility issues with existing rasterizer codebases. As a result, researchers spending significant time resolving linker errors, adjusting makefiles, or ensuring cross-platform consistencyâefforts that contribute little to the scientific goals of their work.

To address these challenges, we designed a flexible module system leveraging PyTorchâs built-in loading mechanisms and extension frameworks [34]. This system supports both Just-in-Time (JIT) loading and Ahead-of-Time (AOT) compilation, enabling seamless integration of custom components with minimal configuration. For JIT loading, modules are compiled dynamically at runtime using PyTorchâs C++ extension APIs. For AOT compilation, modules are built during installation, generating platform-specific binaries that can be loaded instantly at runtime.

```python
Listing 1 Snippet used to register task.
class GaussainModel:
2 def register_pre_task(
3 self,
4 scheduler: Scheduler,
5 ppl: PipelineParams,
6 opt: OptimizationParams
7 ):
8 scheduler.register_task(
9 range(opt.iterations),
10 task=self.update_learning_rate
11 )
```

Built on this foundation, we have integrated distinct rasterizers and a suite of CUDA/C++ modules. Critically, this system abstracts all compilation details from end users: regardless of whether a module is JIT-loaded or pre-built, researchers can import and use it via simple Python import. The underlying framework handles dependency resolution, and kernel loading, ensuring that even complex combinations of rasterizers and compression modules work out-ofthe-box.

## 3.2. Unified Pipeline

Another characteristic of 3DGS is that its training process involves many operations beyond parameter updates and gradient computations [23, 26]. The most representative of these are the pruning and densification of Gaussian points. Additionally, many methods incorporate operations that are coupled with training epochs. For example, in HAC [8], the training is mainly divided into full-precision training, training with simulated quantization, and training with entropy constraints. Beyond these three main training stages, there are also brief pauses in pruning/densification operations. The training phases in CAT-3DGS [48] are even more complex. Manually controlling these intricate training processes solely through if-else code is both cumbersome and error-prone. Therefore, we have implemented a unified training workflow based on a scheduler mechanism. A general flowchart can be referred to in Fig 1.

We have unified the training process of 3DGS into five stages: pre-scheduler execution, rendering, loss calculation, post-scheduler execution, and optimizer update. The scheduler is designed to execute specified tasks at designated training iterations, and Listing 1 demonstrates the API for task registration in the scheduler. When the scheduler executes tasks, it automatically uses appropriate parameters for dispatch based on the type annotations [9] of the taskâs formal parameters. With this design, various operations beyond rendering and loss calculation can be converted into tasks and clearly presented through the scheduling flow. Taking the training process of the original Gaussian as an example, Table 4 shows the tasks included in the pre-scheduler and post-scheduler. One constraint in task allocation is that tasks requiring the use of rendering results must be registered in the post-scheduler, as the prescheduler cannot access rendering results.

<!-- image-->

Figure 1. Framework of Splatwizard. We have abstracted the Gaussian training process into a universal training pipeline, which can support various custom operations through a task scheduler mechanism. The horizontal arrow lines in the diagram illustrate the modules used at different stages of Gaussian training. For instance, different models employ different adaptive density control mechanisms, and such control mechanisms can be registered into the scheduler. This results in a GS model implemented based on Splatwizard, while also seamlessly leveraging various auxiliary modules and functions provided by the framework.
<table><tr><td>Stage</td><td>Task</td><td>Plan</td></tr><tr><td rowspan="2">Pre</td><td>update lr</td><td>range(30000)</td></tr><tr><td>increase SH degree</td><td>range(0, 30000, 1000)</td></tr><tr><td rowspan="2">Post</td><td>collect gradient statistics</td><td>range(15000)</td></tr><tr><td>prune and densify</td><td>range(500, 15000, 100)</td></tr><tr><td></td><td>reset opacity</td><td>range(0, 15000, 3000)</td></tr></table>

Table 2. Schedule plan for vanilla 3DGS. We use python style range expression to illustrate detail execution time.

In addition to clarifying the training workflow, another advantage of the scheduler-based mechanism is its similarity in code organization to the original 3DGS implementation. This allows for easy adaptation of various new works based on the original codebase into a scheduler-based organizational structure. This significantly reduces the difficulty of integrating new methods into our framework.

## 3.3. Function Library

The Rasterizer Lib is a collection of encapsulated rasterizer from previous works. This includes a standard rasterizer [23], along with its subsequent accelerated version [30] and an additional variant that incorporates depth computation [24]. Since pruning methods often rely on importance assessment [18], the framework provides rasterizers capable of calculating global importance scores based on opacity [11], as well as variants that support computing reparameterized approximate Hessian importance [17]. Additionally, it includes versions designed specifically for reconstruction, such as 2DGS [21] and TrimGS [10]. The architectural design of Splatwizard allows for flexible combination of these modules, enabling the mixed use of different rasterizers within a single method.

The Metrics Lib module focuses on loss calculation and performance metric evaluation, providing rich tool support for the Loss module. It includes the implementation of various loss functions and the calculation of image quality metrics (PSNR, SSIM [39] and LPIPS [51]). In addition to these image quality assessment metrics, we have specifically introduced geometric evaluation metrics, including Chamfer distance of point cloud [10] and reconstructed mesh [21]. While such metrics are often unreported in previous compression models, we believe their inclusion can provide a more comprehensive reflection of model performance.

We have categorized modules related to entropy estimation, quantization, and entropy encoding into the Codec Lib. Current research on 3DGS compression has applied a variety of entropy models and encoding algorithms. In terms of quantization, methods such as STE quantization, noise relaxation quantization, vector quantization, and trainingaware quantization are included. For context modeling, models based on hash grids and triplanes have also been utilized. As for entropy encoders, techniques such as arithmetic coding, rANS coding, and Huffman coding are covered. These diverse methods were previously scattered across different studies, and consolidating these modules here will facilitate future research.

Reconstruction Lib acts as a critical bridge between 3DGS and mesh-based models, converting optimized Gaussian primitives into structured meshes. Following previous work, we render depth maps of the training views and utilize truncated signed distance fusion (TSDF) to fuse the reconstruction depth maps, using Open3D [52].

## 3.4. Model Zoo

Building on the aforementioned modular framework, we have systematically reproduced several mainstream algorithms in 3DGS compression, forming a unified benchmark implementation. Throughout the reproduction process, we adhered principle of minimizing reproduction discrepancies: on one hand, we precisely aligned key parameter settings described in the original papers to ensure that the critical logic of each algorithm remains consistent with the original descriptions; on the other hand, leveraging the frameworkâs modular design, we abstracted common components across different algorithms into shared modules, only implementing unique innovations of each algorithm. This approach reduce performance deviations caused by differences in engineering implementation. Notably, during the reproduction process, we identified and fixed several potential issues in the original code that affected result consistency. Through this approach, our implementation not only provides a unified comparison benchmark for 3DGS compression algorithms but also offers a more robust code reference for the community, effectively reducing discrepancies in research caused by differences in engineering implementations.

## 4. ChimeraGS

As a demonstration, we have developed a new model with closer comprehensive performance comparing with end-toend trained model like HAC [8] by strategically integrating effective modules and proposed distillation guided SH pruning. Fig. 2 illustrate the framework of our method.

## 4.1. Pruning Strategy

The original LightGaussian [11] comprises multiple stages: pruning, distillation, and the final vector quantization encoding phase. Each of these stages has room for optimization. During the pruning stage, we can adopt more aggressive pruning methods and ratios to reduce the number of Gaussian primitives. For example, we can adopt the pruning strategy from PUP-3DGS [18] to significantly reduce the number of Gaussian kernels after pruning while maintaining comparable rendering quality. Additionally, beyond post-processing pruning, we can directly learn a compact Gaussian representation to replace the pruning stage. This approach often achieves higher compression rates.

We adopt the SpeedySplat [17] to obtain a highly compact initialization of Gaussians. SpeedySplat utilizes the reparameterized approximation of Hessian matrix to measure the importance of each Gaussian primitive

$$
\tilde { U } _ { i } = \log | \nabla _ { g _ { i } } I _ { \mathcal { G } } \nabla _ { g _ { i } } I _ { \mathcal { G } } ^ { T } | ,\tag{1}
$$

where $I _ { \mathcal { G } }$ is rendered image. $g _ { i }$ is the 2D projected value of $G _ { i }$ at pixel $p ,$ given by

$$
g _ { i } = e ^ { q } , q = - \frac { 1 } { 2 } ( p - \mu _ { i _ { 2 D } } ) { \bf { \Sigma } } _ { i _ { 2 D } } ^ { - 1 } ( p - \mu _ { i _ { 2 D } } ) ^ { T } .\tag{2}
$$

Since log is monotonically increasing, Equation 3 can be further rewrited as

$$
{ \tilde { U } } _ { i } = ( \nabla _ { g _ { i } } I _ { \mathcal { G } } ) ^ { 2 } .\tag{3}
$$

Based on this efficient importance evaluation metric, effective pruning was achieved during the training phase.

## 4.2. Distillation Guided SH Pruning

LightGaussian further reduces the number of parameters per Gaussian primitive by directly reduce the spherical harmonic degree and then applying distillation fine-tuning. This approach significantly reduces the encoded size. However, adopting a more aggressive strategy during the pruning stage may result in a scenario where simply reducing the order followed by fine-tuning fails to adequately compensate for the quality loss caused by the degree reduction. Therefore, we have adopted a dynamic pruning approach here, which determines which Gaussian primitives undergo order reduction in a learnable manner.

More specifically, We apply binary masks $M \in \{ 0 , 1 \} ^ { N }$ to degree-3 SH parameter of each Gaussian primitive.

$$
\hat { \mathbf { c } } _ { n } ^ { ( 3 ) } = M _ { n } \mathbf { c } _ { n } ^ { ( 3 ) } ,\tag{4}
$$

where n is the index of the primitive, $\mathbf { c } _ { n } ^ { ( 3 ) }$ is degree-3 SH parameter. During rendering, we use $\hat { \mathbf { c } } _ { n } ^ { ( 3 ) }$ instead of $\mathbf { c } _ { n } ^ { ( 3 ) }$

Obviously, binary mask M is non-differentiable, following previous works, we use a straight-through-estimator

$$
M _ { n } = \operatorname { s g } ( \mathbb { 1 } [ \sigma ( m _ { n } ) > c ] - \sigma ( m _ { n } ) ) + \sigma ( m _ { n } ) ,\tag{5}
$$

where sg(Â·) is stop gradient operator, 1[Â·] and $\sigma ( \cdot )$ are indicator function and sigmoid respectively. To eliminate redundant degree-3 SH parameters, an additional loss is required

$$
\mathcal { L } _ { m } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \sigma ( m _ { n } ) .\tag{6}
$$

<!-- image-->  
Figure 2. Framework of ChimeraGS. Since Splatwizard natively supports flexible module composition, we can easily mix multiple different rasterizers in same model. The labels on the left side indicate the rasterizers used in each phase.

The use of learnable masks for pruning Gaussians or Gaussian parameters has also been explored in previous works [26, 36]. Our main improvement here is to leverage distillation to further enhance performance. Following LightGaussian [11], we use a teacher model to generate pseudo view to supervise student model

$$
\mathcal { L } _ { \mathrm { d i s t i l l } } = \frac { 1 } { H W } \left\| I _ { \mathrm { t e a c h e r } } ( \mathbf { R } | \mathbf { t } ) - I _ { \mathrm { s t u d e n t } } ( \mathbf { R } | \mathbf { t } ) \right\| _ { 2 } ^ { 2 } ,\tag{7}
$$

where $I _ { \mathrm { t e a c h e r } } ( \mathbf { R } | \mathbf { t } )$ and $I _ { \mathrm { s t u d e n t } } ( \mathbf { R } | \mathbf { t } )$ are the rendered image given a camera rotation R and position t of teacher model and student model respectively. The position is given by

$$
\mathbf { t } _ { \mathrm { p s e u d o } } = \mathbf { t } _ { \mathrm { t r a i n } } + \mathcal { N } ( 0 , \sigma ^ { 2 } ) ,\tag{8}
$$

where $\mathbf { t } _ { \mathrm { t r a i n } }$ is sampled position, $\mathcal { N }$ is denotes a Gaussian distribution with mean 0 and variance $\sigma ^ { 2 }$ . The final loss in distillation stage is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { d i s t i l l } } + \lambda \mathcal { L } _ { m }\tag{9}
$$

By comprehensively applying learnable masks and knowledge distillation, our method can mitigate the performance loss caused by pruning, thereby improving the ratedistortion performance of the model.

## 4.3. Encoding and Rendering

In the encoding phase, we adopt the vector quantization encoder from LightGaussian. The main difference is that we use separate codebooks for the degree-2 and degree-3 SH parameters respectively. Since Gaussian rendering is position-order-independent, Gaussian points can be encoded separately according to their SH degrees. Additionally, to further improve encoding performance, we sort the Gaussian points using Morton order before the final encoding.

Since our method does not rely on a specially implemented rasterizer during the rendering phase, it can be easily integrated with existing rendering acceleration schemes. Here, we have experimented with FlashGS [12] and TC-GS [29] respectively. Combined with the model size advantage brought by compression itself, both methods have achieved impressive rendering speed.

## 5. Benchmarking

## 5.1. Benchmarking Setup

Our evaluation is conducted on six multi-view image datasets: Mip-NeRF 360 [4], Tanks & Temples [25], Deep Blending [19], BungeeNeRF [41], NeRF Synthetic [31] and DTU. Tanks&Temples features real-world scenes with complex geometry captured using laser scanners, while Mip-NeRF 360 contains 9 unbounded scenes enabling 360- degree rendering. Deep Blending provides diverse indoor and outdoor environments, BungeeNeRF focuses on extreme multi-scale scenarios, and NeRF Synthetic offers 8 photorealistic Blender-rendered scenes. For geometric evaluation, DTU dataset [22] provides measure based geometry ground truth. This enables us to evaluate the reconstruction accuracy of the model, thereby providing a more comprehensive reflection of its performance.

In terms of model selection, we evaluated all methods in the model zoo, with specific methods detailed in the Table 1. All approaches were implemented or reproduced based on SW, using unified evaluation protocols and consistent training/testing dataset splits. Furthermore, to prevent potential data leakage from affecting bitrate evaluation, we strictly isolated the models used during the encoding and decoding phases. All included methods support encoding complete information into a single binary file, while the decoding phase utilizes only a newly initialized model and the corresponding binary file for testing.

<!-- image-->  
Figure 3. Results of Mip-NeRF 360. Note that when counting the total number of Gaussian primitives in HAC and CAT-3DGS, only the number of anchor is considered, and the values are not directly comparable to other methods, hence they are drawn with a dashed line.

## 5.2. Photorealistic Metrics Results

Fig. 3 presents evaluation results of all methods on Mip-NeRF, with results on additional datasets included in the Supplementary. Beyond the previously mentioned PSNR, SSIM, and LPIPS metrics, we also measured bitrate, Gaussian count, inference rendering speed, and peak memory usage during rendering. The figure reports cross-comparison results of these metrics.

From the conventional rate-distortion curves, we observe a trend: methods employing hybrid implicit neural networks and 3DGS generally achieve superior rate-distortion performance, while models that combine quantization operations with rate-distortion optimization also demonstrate relatively competitive performance. However, when considering Gaussian count as the metric for bitrate, the aforementioned observations may not necessarily hold.

<!-- image-->  
Figure 4. Results of DTU dataset. Here we present the PSNR, along with the Chamfer distance based on mesh and point cloud respectively.

Furthermore, comparisons of rendering frame rate and peak GPU memory usage provide a more comprehensive perspective. As expected, hybrid representation-based models generally exhibit lower frame rates compared to pruning-based approaches, though their impact on memory consumption is not significant.

The proposed ChimeraGS also achieves competitive results across several dimensions. Compared to prior approaches, our method attains rate-distortion performance closer to HAC [8], while achieving top-tier results in terms of Gaussian count. However, experimental observations reveal that multi-stage models like LightGaussian [11] incur additional performance overhead between stages. Combined with previous findings, this insight points toward future improvement directions, such as incorporating quantization during training to enhance performance.

## 5.3. Geometric Metrics Results

Fig. 4 shows the rate-distortion performance of geometric metrics. Unlike the previous section, the evaluation for photo employs the novel view synthesis task, where the dataset is divided into training and test sets, and the decoded model is only tested on the test set for reconstruction metrics. In the measurement of geometric metrics, we primarily focus on geometric reconstruction accuracy; therefore, the entire dataset is used for training. Correspondingly, the ratedistortion curves here only represent data fitting capability, not novel view synthesis capability. Since current TSDFbased mesh reconstruction methods rely on depth maps derived from Gaussians, for all models using standard rasterizers (including hybrid models), we have replaced their rasterizers with versions that support depth computation to facilitate subsequent mesh reconstruction stages.

Compared to photo-level evaluation, we find that some methods even achieve point cloud-level accuracy that approaches or surpasses the performance of 2DGS. This suggests that compression tasks and geometric reconstruction tasks are not entirely conflicting objectives. In the future, we could explore integrating methods from these two specialized tasks to investigate pathways for enhancing each otherâs performance. Meanwhile, we should also note that not every compressor can improve geometric quality, so the design of specific methods remains particularly important.

## 6. Conclusion and Future Work

In this paper, we introduce a universal framework for GS compression evaluation, named Splatwizard. This framework provides a unified training and evaluation pipeline adaptable to various Gaussian methods, while offering a flexible module utilization mechanism and a comprehensive function library. Based on this framework, we have reproduced multiple Gaussian compression methods and, leveraging the flexible architecture, developed a distinctive approach with unique performance characteristics. Experimental results include multi-dimensional comparisons of all methods. We hope this work will accelerate research in this field by providing fundamental tools and resources. In the future, we will further expand the capabilities of this framework by incorporating a wider range of methods for comparison, while systematically organizing its functional modules to facilitate their use in subsequent work. Additionally, while the current framework is primarily tailored for static GS scene compression methods, we plan to extend its functionality to tasks like dynamic scenes [40, 43], feed-forward paradigm [7] and streaming 3DGS compression [35].

## References

[1] Muhammad Salman Ali, Maryam Qamar, Sung-Ho Bae, and Enzo Tartaglione. Trimming the fat: Efficient compression of 3d gaussian splats through pruning. arXiv preprint arXiv:2406.18214, 2024. 2

[2] M. T. Bagdasarian, P. Knoll, Y. Li, F. Barthel, A. Hilsmann, P. Eisert, and W. Morgenstern. 3dgs.zip: A survey on 3d gaussian splatting compression methods. Computer Graphics Forum, page e70078, 2025. https://wm.github.io/3dgs-compression-survey/. 1

[3] Milena T Bagdasarian, Paul Knoll, Y Li, Florian Barthel, Anna Hilsmann, Peter Eisert, and Wieland Morgenstern. 3dgs. zip: A survey on 3d gaussian splatting compression methods. In Computer Graphics Forum, page e70078. Wiley Online Library, 2025. 3

[4] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. CVPR, 2022. 6

[5] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased grid-based neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19697â19705, 2023. 2

[6] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, and Guofeng Zhang. Pgsr: Planar-based gaussian splatting for efficient and high-fidelity surface reconstruction. arXiv preprint arXiv:2406.06521, 2024. 1

[7] Yihang Chen, Qianyi Wu, Mengyao Li, Weiyao Lin, Mehrtash Harandi, and Jianfei Cai. Fast feedforward 3d gaussian splatting compression. arXiv preprint arXiv:2410.08017, 2024. 8

[8] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi, and Jianfei Cai. Hac: Hash-grid assisted context for 3d gaussian splatting compression. In European Conference on Computer Vision, pages 422â438. Springer, 2024. 2, 3, 5, 8

[9] Luca Di Grazia and Michael Pradel. The evolution of type annotations in python: an empirical study. In Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, pages 209â220, 2022. 3

[10] Lue Fan, Yuxue Yang, Minxing Li, Hongsheng Li, and Zhaoxiang Zhang. Trim 3d gaussian splatting for accurate geometry representation. arXiv preprint arXiv:2406.07499, 2024. 4

[11] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, Zhangyang Wang, et al. Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps. Advances in neural information processing systems, 37: 140138â140158, 2024. 2, 3, 4, 5, 6, 8

[12] Guofeng Feng, Siyan Chen, Rong Fu, Zimu Liao, Yi Wang, Tao Liu, Boni Hu, Linning Xu, Zhilin Pei, Hengjie Li, et al. Flashgs: Efficient 3d gaussian splatting for large-scale and high-resolution rendering. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26652â 26662, 2025. 3, 6

[13] John Flynn, Michael Broxton, Lukas Murmann, Lucy Chai, Matthew DuVall, Clement Godard, Kathryn Heal, Srinivas Â´ Kaza, Stephen Lombardi, Xuan Luo, et al. Quark: Real-time, high-resolution, and general neural view synthesis. ACM Transactions on Graphics (TOG), 43(6):1â20, 2024. 2

[14] Yongjie Guan, Xueyu Hou, Nan Wu, Bo Han, and Tao Han. Metastream: Live volumetric content capture, creation, delivery, and rendering in real time. In Proceedings of the 29th Annual International Conference on Mobile Computing and Networking, New York, NY, USA, 2023. Association for Computing Machinery. 1

[15] Minghao Guo, Bohan Wang, Kaiming He, and Wojciech Matusik. Tetsphere splatting: Representing high-quality geometry with lagrangian volumetric meshes. arXiv preprint arXiv:2405.20283, 2024. 1

[16] Xiaoshen Han, Minghuan Liu, Yilun Chen, Junqiu Yu, Xiaoyang Lyu, Yang Tian, Bolun Wang, Weinan Zhang, and Jiangmiao Pang. Re3sim: Generating high-fidelity simulation data via 3d-photorealistic real-to-sim for robotic manipulation. arXiv preprint arXiv:2502.08645, 2025. 1

[17] Alex Hanson, Allen Tu, Geng Lin, Vasu Singla, Matthias Zwicker, and Tom Goldstein. Speedy-splat: Fast 3d gaussian splatting with sparse pixels and sparse primitives. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21537â21546, 2025. 2, 4, 5

[18] Alex Hanson, Allen Tu, Vasu Singla, Mayuka Jayawardhana, Matthias Zwicker, and Tom Goldstein. Pup 3d-gs: Principled uncertainty pruning for 3d gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5949â5958, 2025. 2, 4, 5

[19] Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm, George Drettakis, and Gabriel Brostow. Deep blending for free-viewpoint image-based rendering. ACM Trans. Graph., 37(6), 2018. 6

[20] Jan Held, Renaud Vandeghen, Sanghyun Son, Daniel Rebain, Matheus Gadelha, Yi Zhou, Ming C. Lin, Marc Van Droogenbroeck, and Andrea Tagliasacchi. Triangle splatting+: Differentiable rendering with opaque triangles, 2025. 1

[21] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 conference papers, pages 1â11, 2024. 1, 3, 4

[22] Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola, and Henrik AanÃ¦s. Large scale multi-view stereopsis evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 406â413, 2014. 6

[23] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023. 1, 2, 3, 4

[24] Bernhard Kerbl, Andreas Meuleman, Georgios Kopanas, Michael Wimmer, Alexandre Lanvin, and George Drettakis. A hierarchical 3d gaussian representation for real-time rendering of very large datasets. ACM Transactions on Graphics (TOG), 43(4):1â15, 2024. 4

[25] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene

reconstruction. ACM Transactions on Graphics, 36(4), 2017. 6

[26] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park. Compact 3d gaussian representation for radiance field. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21719â 21728, 2024. 2, 3, 6

[27] Ke Li, Reinhard Bacher, Susanne Schmidt, Wim Leemans, and Frank Steinicke. Reality fusion: Robust real-time immersive mobile robot teleoperation with volumetric visual data fusion. arXiv preprint arXiv:2408.01225, 2024. 1

[28] Sicheng Li, Chengzhen Wu, Hao Li, Xiang Gao, Yiyi Liao, and Lu Yu. Gscodec studio: A modular framework for gaussian splat compression. arXiv preprint arXiv:2506.01822, 2025. 3

[29] Zimu Liao, Jifeng Ding, Siwei Cui, Ruixuan Gong, Boni Hu, Yi Wang, Hengjie Li, XIngcheng Zhang, Hui Wang, and Rong Fu. Tc-gs: A faster gaussian splatting module utilizing tensor cores. arXiv preprint arXiv:2505.24796, 2025. 3, 6

[30] Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl, Markus Steinberger, Francisco Vicente Carrasco, and Fernando De La Torre. Taming 3dgs: High-quality radiance fields with limited resources. In SIGGRAPH Asia 2024 Conference Papers, pages 1â11, 2024. 4

[31] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In Proceedings of the European Conference on Computer Vision (ECCV), 2020. 6

[32] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 2

[33] Simon Niedermayr, Josef Stumpfegger, and Rudiger West- Â¨ ermann. Compressed 3d gaussian splatting for accelerated novel view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10349â10358, 2024. 2

[34] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019. 3

[35] Jiakai Sun, Han Jiao, Guangyuan Li, Zhanjie Zhang, Lei Zhao, and Wei Xing. 3dgstream: On-the-fly training of 3d gaussians for efficient streaming of photo-realistic freeviewpoint videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20675â20685, 2024. 8

[36] Henan Wang, Hanxin Zhu, Tianyu He, Runsen Feng, Jiajun Deng, Jiang Bian, and Zhibo Chen. End-to-end ratedistortion optimized 3d gaussian representation. In European Conference on Computer Vision, pages 76â92. Springer, 2024. 6

[37] Penghao Wang, Zhirui Zhang, Liao Wang, Kaixin Yao, Siyuan Xie, Jingyi Yu, Minye Wu, and Lan Xu. VË3: View-

ing volumetric videos on mobiles via streamable 2d dynamic gaussians. CoRR, abs/2409.13648, 2024. 1

[38] Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P Srinivasan, Howard Zhou, Jonathan T Barron, Ricardo Martin-Brualla, Noah Snavely, and Thomas Funkhouser. Ibrnet: Learning multi-view image-based rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4690â4699, 2021. 2

[39] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600â612, 2004. 4

[40] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20310â20320, 2024. 8

[41] Yuanbo Xiangli, Linning Xu, Xingang Pan, Nanxuan Zhao, Anyi Rao, Christian Theobalt, Bo Dai, and Dahua Lin. Bungeenerf: Progressive neural radiance field for extreme multi-scale scene rendering. In The European Conference on Computer Vision (ECCV), 2022. 6

[42] Shuzhao Xie, Weixiang Zhang, Chen Tang, Yunpeng Bai, Rongwei Lu, Shijia Ge, and Zhi Wang. Mesongs: Posttraining compression of 3d gaussians via efficient attribute transformation. In European Conference on Computer Vision. Springer, 2024. 2

[43] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for highfidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20331â20341, 2024. 8

[44] Yuxuan Yao, Zixuan Zeng, Chun Gu, Xiatian Zhu, and Li Zhang. Reflective gaussian splatting. arXiv preprint arXiv:2412.19282, 2024. 2

[45] Chongjie Ye, Yinyu Nie, Jiahao Chang, Yuantao Chen, Yihao Zhi, and Xiaoguang Han. Gaustudio: A modular framework for 3d gaussian splatting and beyond. arXiv preprint arXiv:2403.19632, 2024. 1, 2

[46] Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, et al. gsplat: An open-source library for gaussian splatting. Journal of Machine Learning Research, 26(34):1â17, 2025. 2

[47] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19447â19456, 2024. 2

[48] Yu-Ting Zhan, Cheng-Yuan Ho, Hebi Yang, Yi-Hsin Chen, Jui Chiu Chiang, Yu-Lun Liu, and Wen-Hsiao Peng. Cat-3dgs: A context-adaptive triplane approach to ratedistortion-optimized 3dgs compression. arXiv preprint arXiv:2503.00357, 2025. 2, 3

[49] Fengdi Zhang, Hongkun Cao, and Ruqi Huang. Consistent quantity-quality control across scenes for deployment-aware gaussian splatting. arXiv preprint arXiv:2505.10473, 2025. 2

[50] Jiakai Zhang, Liao Wang, Xinhang Liu, Fuqiang Zhao, Minzhang Li, Haizhao Dai, Boyuan Zhang, Wei Yang, Lan Xu, and Jingyi Yu. Neuvv: Neural volumetric videos with immersive rendering and editing. CoRR, abs/2202.06088, 2022. 1

[51] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586â595, 2018. 4

[52] Qian-Yi Zhou, Jaesik Park, and Vladlen Koltun. Open3d: A modern library for 3d data processing. arXiv preprint arXiv:1801.09847, 2018. 5

# Splatwizard: A Benchmark Toolkit for 3D Gaussian Splatting Compression

Supplementary Material

## 7. Schedule plan of methods

In this section, we will introduce the execution plans for all methods included. We use python style range expression to illustrate detail execution time.

## 7.1. 3DGS

<table><tr><td>Stage</td><td>Task</td><td>Plan</td></tr><tr><td rowspan="2">Pre</td><td>update lr</td><td>range(30000)</td></tr><tr><td>increase SH degree</td><td>range(0, 30000, 1000)</td></tr><tr><td rowspan="3">Post</td><td>collect gradient statistics</td><td>range(15000)</td></tr><tr><td>prune and densify</td><td>range(500, 15000, 100)</td></tr><tr><td>reset opacity</td><td>range(0, 15000, 3000)</td></tr></table>

Table 3. Schedule plan for vanilla 3DGS.

## 7.5. LightGaussian

Note we separate prune and distillation to different stages.

<table><tr><td>Stage</td><td>Task</td><td>Plan</td></tr><tr><td>Post</td><td>prune</td><td>1</td></tr></table>

Table 7. Schedule plan for LightGaussian PRUNE stage.

<table><tr><td>Stage</td><td>Task</td><td>Plan</td></tr><tr><td>Post</td><td>calculate importance score</td><td>5000</td></tr></table>

Table 8. Schedule plan for LightGaussian DISTILL stage.

## 7.2. Trimming the Fat

<table><tr><td>Stage</td><td>Task</td><td>Plan</td></tr><tr><td>Pre</td><td>update lr</td><td>range(30000)</td></tr><tr><td>Post</td><td>gradient aware prune</td><td>range(100, 5000, 500)</td></tr></table>

Table 4. Schedule plan for Trimming the Fat.

## 7.6. MesonGS

<table><tr><td>Stage</td><td>Task</td><td>Plan</td></tr><tr><td rowspan="3">Pre</td><td>update lr</td><td>range(3000)</td></tr><tr><td>calculate importance score</td><td>1</td></tr><tr><td>create octree</td><td>1</td></tr></table>

Table 9. Schedule plan for MesonGS.

## 7.3. PUP-3DGS

<table><tr><td>Stage</td><td>Task</td><td>Plan</td></tr><tr><td>Pre</td><td>update lr</td><td>range(30000)</td></tr><tr><td>Post</td><td>gradient aware prune</td><td>range(100, 5000, 500)</td></tr></table>

Table 5. Schedule plan for PUP-3DGS.

## 7.7. C3DGS

<table><tr><td>Stage</td><td>Task</td><td>Plan</td></tr><tr><td>Pre</td><td>update lr</td><td>range(30000)</td></tr><tr><td></td><td>VQ compress</td><td>1</td></tr></table>

Table 10. Schedule plan for C3DGS.

## 7.4. Speedy-Splat

<table><tr><td>Stage</td><td>Task</td><td>Plan</td></tr><tr><td>Pre</td><td>update lr increase SH degree</td><td>range(30000) range(0, 30000, 1000)</td></tr><tr><td>Post</td><td>collect gradient statistics</td><td>range(15000)</td></tr><tr><td></td><td>prune and densify</td><td>range(500, 15000, 100)</td></tr><tr><td></td><td>reset opacity</td><td>range(0, 15000, 3000)</td></tr><tr><td></td><td>soft prune</td><td>range(6000, 15000, 3000)</td></tr><tr><td></td><td>hard prune</td><td>range(15000, 30000, 3000)</td></tr></table>

Table 6. Schedule plan for Speedy-Splat.

## 7.8. Compact3DGS

<table><tr><td>Stage</td><td>Task</td><td>Plan</td></tr><tr><td rowspan="3">Pre</td><td>update lr</td><td>range(30000)</td></tr><tr><td>increase SH degree</td><td>range(0, 30000, 1000)</td></tr><tr><td>switch to RVQ training</td><td>29000</td></tr><tr><td rowspan="4">Post</td><td>collect gradient statistics</td><td>range(15000)</td></tr><tr><td>prune and densify</td><td>range(500, 15000, 100)</td></tr><tr><td>reset opacity</td><td>range(0, 15000, 3000)</td></tr><tr><td>mask prune</td><td>range(15000, 30000, 1000)</td></tr></table>

Table 11. Schedule plan for Compact3DGS.

## 7.9. HAC

<table><tr><td>Stage</td><td>Task</td><td>Plan</td></tr><tr><td>Pre</td><td>update lr</td><td>range(30000)</td></tr><tr><td>Post</td><td>update anchor bound</td><td>1</td></tr><tr><td></td><td>collect gradient statistics</td><td>range(15000) range(1500, 3000, 100)</td></tr><tr><td></td><td>adjust anchor adjust anchor</td><td>range(4000, 15000, 100)</td></tr><tr><td></td><td>switch to quantized training</td><td>3000</td></tr><tr><td></td><td>switch to entropy training</td><td>10001</td></tr></table>

Table 12. Schedule plan for HAC.

## 7.10. CAT-3DGS

<table><tr><td>Stage</td><td>Task</td><td>Plan</td></tr><tr><td>Pre</td><td>update lr</td><td>range(30000)</td></tr><tr><td></td><td>update cam mask</td><td>range(30000)</td></tr><tr><td></td><td>update anchor bound</td><td>1</td></tr><tr><td></td><td>setup triplane</td><td>10000</td></tr><tr><td></td><td>switch to quantized training</td><td>3000</td></tr><tr><td></td><td>switch to entropy training</td><td>10001</td></tr><tr><td>Post</td><td>collect gradient statistics</td><td>range(15000)</td></tr><tr><td></td><td>adjust anchor</td><td>range(1500, 3000, 100)</td></tr><tr><td></td><td>adjust anchor</td><td>range(4000, 15000, 100)</td></tr></table>

Table 13. Schedule plan for CAT-3DGS.

## 8. More experiments results.

These figures respectively show the results on the Deep Blending, NeRF Synthetic, and BungeeNeRF datasets. All training and testing were completed on a single RTX 3090. Note that when counting the total number of Gaussian primitives in HAC and CAT-3DGS, only the number of anchor is considered, and the values are not directly comparable to other methods, hence they are drawn with a dashed line.

<!-- image-->  
Figure 5. Results of DeepBlending.

<!-- image-->  
Figure 6. Results of BungeeNeRF.

<!-- image-->  
Figure 7. Results of NeRF Synthetic.

<!-- image-->  
Figure 8. Results of Tanks & Temples.