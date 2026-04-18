# SplatBus: A Gaussian Splatting Viewer Framework via GPU Interprocess Communication

Yinghan Xu Theo MoralesÂ´ John Dingliana Trinity College Dublin Dublin, Ireland

## Abstract

Radiance field-based rendering methods have attracted significant interest from the computer vision and computer graphics communities. They enable high-fidelity rendering with complex real-world lighting effects, but at the cost of high rendering time. 3D Gaussian Splatting solves this issue with a rasterisation-based approach for real-time rendering, enabling applications such as autonomous driving, robotics, virtual reality, and extended reality. However, current 3DGS implementations are difficult to integrate into traditional mesh-based rendering pipelines, which is a common use case for interactive applications and artistic exploration. To address this limitation, this software solution uses Nvidiaâs interprocess communication (IPC) APIs to easily integrate into implementations and allow the results to be viewed in external clients such as Unity, Blender, Unreal Engine, and OpenGL viewers. The code is available at https://github.com/RockyXu66/splatbus.

## 1. Introduction

Conventionally, rasterization relies on mesh-based representations, which require complex lighting models (i.e. physically-based rendering) to reproduce real-world effects. Recently, Neural Radiance Field (NeRF) methods [14, 15, 20] popularized novel view synthesis (NVS) of scenes with complex lighting effects â such as reflections and refractions â from multi-view images. However, NeRFs rely on engineering tricks and highly optimized implementations to enable real-time rendering [15]. To address this limitation, 3D Gaussian Splatting (3DGS) [8] leverages the parallelism and hardware specialization of rasterization. 3DGS achieves this by representing scenes as 3D Gaussian primitives for a discrete radiance field, achieving a favourable balance between rendering quality and efficiency. However, the original 3DGS viewer [1] is implemented as a standalone C++ application tightly coupled with its CUDA-based rasterizer. As a result, it can only render 3D Gaussian primitives and does not support other representations, such as 3D meshes. Moreover, as 3DGS-based rendering methods continue to evolve, many works opt to modify the CUDA rasterizer and adapt the viewer accordingly, thereby introducing significant engineering overhead.

Currently, several solutions exist to integrate 3D Gaussian Splatting into game engines, enabling hybrid rendering with 3D meshes [13, 18, 19, 24]. However, their generalizability is limited. Common limitations include converting 3D Gaussian representations into particle systems, a lack of support for the latest rasterizer variants, or difficulty in deploying intermediate neural network components before rasterization. Other tools support 3D Gaussian Splatting or 2D Gaussian Splatting [4], but they do not meet the requirements of many research projects, such as dynamic radiance fields.

In this project, we aim to generalize the viewing process by decoupling the rasterizer from the viewer. We treat the rasterizer as an independent rendering server, while different viewers act as clients to meet diverse application needs. To enable efficient communication, we use Nvidia interprocess communication (IPC) APIs to share memory pointers and event handles from the server to the client. This design allows the viewer to directly access rendered RGB and depth images, avoiding additional memory copies, reducing latency, and simplifying system integration. With the depth information, we apply depth-aware blending to composite Gaussian-based rendering with mesh-based rendering.

## 2. Related work

## 2.1. Stand-alone viewers

The original 3DGS implementation comes with an interactive viewer based on the System for Image-Based Rendering (SIBR) [5]. Users control the camera via common keyboard and mouse controls. The viewer also supports snapping the camera to training poses. To improve accessibility, some developers use web technologies to render the scene directly in browsers. splat [9] and its video variant splatV [10] adopt WebGL approach for real-time rasterization. In this implementation, the 3D Gaussians are sorted on the CPU and rendered on the GPU, enabling fast rendering of 3D scenes. As one of the early open-source implementations for the Three.js ecosystem, GaussianSplats3D [7] is a foundational framework for rendering 3D Gaussian Splats in standard web browsers and consumer hardware. It introduces optimizations such as WASM-based sorting and a compressed storage format to facilitate the distribution and real-time visualization of large-scale radiance fields. Building upon these works, Spark [23] represents the current state-of-the-art in high-performance web-based 3DGS rendering. It focuses on production-ready features, including GPU-driven shader graphs, cross-device compatibility for mobile WebGL 2.0, and support for dynamic scene elements like skeletal animation and real-time editing. Unlike stand-alone renderers, SuperSplat [18] is a specialized webbased editor built on the PlayCanvas engine [17], designed for the post-processing of Gaussian Splatting data. It enables operations such as spatial cropping, scene alignment, and data compression, serving as a bridge between raw reconstruction outputs and optimized deployment for interactive web applications. Apart from client-side WebGL rendering, NerfStudio [25] adopts a client-server architecture to visualize 3DGS using the gsplat [33] library on the backend to stream real-time, interactive renders to a web-based frontend powered by the Viser [34] framework. Finally, several commercial projects support training 3DGS scenes either on-device or on remote servers, with a web-based viewer, such as LumaAI [13] and Scaniverse [22].

With the recent popularization of dynamic 3DGS methods came the need for dynamic viewers. To the best of our knowledge, EasyVolcap [28] is the only general-purpose viewer and framework for volumetric video capture, which includes point-based methods such as 4D Gaussian Splatting [27, 29, 32]. Although this framework provides the flexibility for research implementation, it imposes a code architecture that is unsuited to most methods that extend the original 3DGS implementation, such as the popular 4DGS [32] and its derivative works [6, 11, 30]. As a result, some of these methods propose their own 4D viewer, either as a modified SIBR [12] or as a stand-alone OpenGL wrapper [31].

Most of the viewer systems mentioned above only support the standard 3DGS rasterization pipeline. As a result, they cannot directly visualize alternative Gaussian-based rasterization methods and often require additional engineering effort to adapt the viewer to modified or extended rasterizers. In addition, these systems do not support hybrid representations, which limits the ability to render 3D meshes together with Gaussian primitives. Our proposed solution addresses this limitation by providing a simple hook into the existing code, which integrates into a client-server ar-

chitecture.

## 2.2. Viewing in game engines

Game engines (e.g., Unity and Unreal Engine) are attractive front-ends for Gaussian Splatting since they provide mature real-time rendering pipelines and interaction/VR support. Engine-based viewers also make it convenient to integrate splats into existing content (meshes, UI), enabling hybrid rendering workflows that are difficult to achieve in standalone research viewers.

Several open-source projects have brought 3D Gaussian Splatting into engines using engine-specific integrations. UnityGaussianSplatting [19] implements the original 3DGS rasterization pipeline inside Unity, utilizing Compute Shaders for efficient GPU-based sorting and rendering. It supports direct 3DGS file import and real-time visualization. GaussianSplattingVRViewerUnity [2] integrates differentiable Gaussian rasterization as a Unity native plugin and achieves OpenXR VR viewing, adding features such as multi-model loading and depth-aware compositing with the Unity scene. On the Unreal side, XScene-UEPlugin [3] provides a UE5 plugin for real-time rendering and editing, emphasizing Niagara-based rendering, importing/converting from original 3DGS assets, and hybrid rendering with other UE content. Despite their usefulness for applications, these integrations are often tightly coupled to a particular renderer implementation and engine runtime, which can limit generalizability for research workflows. Supporting newly modified rasterizers, deploying intermediate neural components before rasterization, or reusing the same renderer across multiple viewers typically requires substantial re-engineering. In contrast, our approach decouples the rasterizer from the viewer by taking the rasterizer as an independent rendering server and using GPU-efficient interprocess communication to share rendered outputs with different engine (and non-engine) clients.

## 2.3. Profilers and monitors

The development of dynamic 3DGS methods â also known as 4D Gaussian Splatting â revealed a lack of tooling for debugging and profiling real-time rendering of volumetric videos. The inherent motion and appearance dynamics of these methods require additional visual information to help in the development process. While 3DGS-based methods are well defined and can be represented by a standardized format, allowing one general-purpose monitor [21] to be used across projects, 4DGS methods suffer from an illposed definition.

Without a standardized 4D representation, each method must implement its data acquisition and plotting specifically within the viewer. To overcome this challenge, we propose to decouple this data acquisition and plotting from the viewer via our client-server architecture. As such, a general-purpose profiler and visual debugger can be implemented as an OpenGL client to our system, where the data acquisition is done in the userâs code base.

## 3. Method

We design a client-server viewer framework to decouple Gaussian Splatting rendering from interactive visualization, improving generalization and easing integration into existing codebases (see Fig. 1). The renderer runs as a Python âserverâ that exposes GPU frame buffers and receives lightweight interaction commands, while multiple âclientsâ (Unity plugin, Blender plugin, and an OpenGL viewer) visualize the output and provide user controls. All clients share the same communication protocol, enabling cross-application integration without modifying the core rasterizer. In addition, this architecture enables user-side data acquisition for profilers and visual debuggers.

## 3.1. SplatBus (Python server package)

SplatBus is a lightweight Python package that can be installed on top of an existing Gaussian-based rasterization environment. On the server side, it exposes two CUDA frame buffers, a color buffer (RGBA32F) and a depth buffer (R32F), managed by the CUDAFrameBuffer class. Each buffer is allocated as contiguous CUDA memory and updated directly on GPU every rendering iteration. For depth, the server converts inverse depth (commonly produced by Gaussian splatting, 1/z) into linear depth (z) to match the client-side consumption. To share these buffers across processes, IPCHandleManager creates CUDA IPC memory handles for both buffers and an IPC-capable CUDA event handle. After writing a new frame into the buffers, the server records the CUDA event to signal completion, allowing clients to synchronize reads without stalling the renderer. Communication uses two TCP socket channels (length-prefixed JSON) for simplicity and cross platform support. The IPC channel is used for one-time initialization: when a client connects, the server sends an init packet containing buffer metadata (resolution, format, pitch, device pointers) and the base64-encoded CUDA IPC memory/event handles needed to open the shared resources. The message channel is used for runtime interaction: the client streams scene-control messages, currently including camera pose and point-cloud (Gaussians) pose. The server converts client poses (e.g., Unity conventions) into the Gaussian Splatting view representation and applies them to update the renderer state inside the userâs rendering loop.

## 3.2. Client architecture

All three clients follow the same two-step workflow: (i) connect to the IPC channel, receive the init packet, and open the shared CUDA buffers using the provided IPC handles; (ii) connect to the message channel to stream interaction commands (camera/object transforms). They differ only in how the received GPU buffers are integrated into each runtime (Unity rendering pipeline, Blender viewport integration, or OpenGL texture upload/interop), while the serverside protocol and renderer integration remain identical.

## 3.2.1. Unity plugin

The Unity client is implemented as a native rendering plugin, extending Unityâs official NativeRenderingPlugin sample [26]. The architecture consists of a C++ native plugin and a C# managed interface. The native plugin runs a background thread that establishes a TCP connection to Gaussian Splatting rasterization server and receives CUDA IPC handles. These handles include cudaIpcMemHandle t structures for shared GPU memory buffers (color in RGBA32F format and depth in R32F format) and a cudaIpcEventHandle t for crossprocess synchronization.

To transfer data from shared CUDA memory into Unityâs rendering pipeline, the plugin relies on OpenGLâCUDA interoperability. Unity RenderTextures are registered with CUDA using cudaGraphicsGLRegisterImage. For each frame, the shared CUDA buffers are copied directly into the corresponding OpenGL texture arrays by cudaMemcpy2DToArrayAsync, using an asynchronous device-to-device transfer. From the CPUâs perspective, this leads to a zero-copy data path, as all data movement occurs entirely on the GPU.

In addition to receiving rendered frames, the Unity plugin sends camera and scene parameters to the server through a separate TCP channel. The communication channels are implemented in a C# component for easier maintenance. Specifically, the plugin streams the camera position and orientation. When point clouds extracted from Gaussian assets are available, they can be imported into the Unity scene and manipulated interactively in the editor view, while the corresponding poses are streamed to the server for rendering.

Figure 2 demonstrates a trained 3DGS scene rendered using the original 3DGS project, with our SplatBus as the server and the Unity plugin as the client. Point clouds are extracted from the 3DGS representation and imported into the Unity scene for interactive control. Additional 3D sphere and cube meshes are integrated into the same 3DGS scene.

Figure 3 presents rendering results from a Gaussian avatar reconstruction method MMLPHuman [35] that uses spatially distributed MLPs to model pose-dependent, highfrequency appearance details of a human avatar. Visualizing such 3DGS-based methods in Unity is challenging because neural network components are often involved before rasterization. With our viewer system, users only need to install the SplatBus package on top of the original research codebase, and the results can be viewed directly in external clients without additional engineering effort.

<!-- image-->  
Figure 1. System diagram of our solution. Our server can be easily integrated into any codebase as a set of Python API calls. The camera controls and other arbitrary data are transferred to and from the client (e.g. Unity, an OpenGL application) via sockets, while the rendered frame is transferred via Nvidiaâs Inter-Process Communication (IPC).

<!-- image-->  
Figure 2. Unity plugin visualizing real-time 3D Gaussian Splatting results with interactive point clouds and 3D meshes.

<!-- image-->  
Figure 3. Unity plugin visualizing real-time Gaussian avatar [35] with 3D meshes.

## 3.2.2. OpenGL viewer

As a general-purpose stand-alone viewer application, we propose an OpenGL interactive wrapper with plotting ability via the Dear ImGui [16] library. This application renders the frame as a texture object and is not intended for mesh or other 3D data integration. Our architecture allows seamless transfer of arbitrary data between the research code and the viewer, enabling visual debugging and profiling. The OpenGL viewer comes with default keyboard and mouse controls and is easily extendable to more customized setups.

## 4. Conclusion

We proposed SplatBus, a lightweight viewer framework that decouples Gaussian Splatting rasterization from visualization by sharing GPU color and depth buffers using CUDA IPC. This design enables low-latency integration with external clients (e.g., a Unity native plugin, an OpenGL viewer) without modifying the core rasterizer, while supporting depth-aware compositing with conventional rendering. Overall, SplatBus lowers the engineering cost of deploying new or evolving Gaussian renderers inside interactive applications. Future work includes richer interaction and advanced rendering effects such as shadows and lighting, as well as broader cross-platform deployment and smoother integration across multiple graphics backends.

## References

[1] Sebastien Bonopera, Jerome Esnault, Siddhant Prakash, Simon Rodriguez, Theo Thonat, Mehdi Benadel, Gaurav Chaurasia, Julien Philip, and George Drettakis. sibr: A system for image based rendering, 2020. 1

[2] CLARTE. GaussianSplattingVRViewerUnity: OpenXRbased VR viewer for 3D Gaussian Splatting in Unity. https : / / github . com / clarte53 / GaussianSplattingVRViewerUnity, 2023. 2

[3] XVERSE Engine. XScene-UEPlugin: High-performance 3DGS plugin for Unreal Engine. https://github. com/xverse-engine/XScene-UEPlugin, 2024. 2

[4] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 Conference Papers, New York, NY, USA, 2024. Association for Computing Machinery. 1

[5] Inria. SIBR: System for Image-Based Rendering. https: //sibr.gitlabpages.inria.fr, 2023. 1

[6] Taeho Kang, Jaeyeon Park, Kyungjin Lee, and Youngki Lee. Clustered error correction with grouped 4d gaussian splatting. In SIGGRAPH Asia 2025 Conference Papers, 2025. 2

[7] Mark Kellogg. GaussianSplats3D: Three.js-based implementation of 3d gaussian splatting. https://github. com/mkkellogg/GaussianSplats3D, 2023. 2

[8] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4), 2023. 1

[9] Kevin Kwok. Splat: A WebGL 3D Gaussian Splatting Viewer. https://github.com/antimatter15/ splat, 2023. 1

[10] Kevin Kwok. Splatv: A Video WebGL 3D Gaussian Splatting Viewer. https : / / github . com / antimatter15/splatv, 2024. 2

[11] Junoh Lee, ChangYeon Won, Hyunjun Jung, Inhwan Bae, and Hae-Gon Jeon. Fully explicit dynamic guassian splatting. In Proceedings of the Neural Information Processing Systems, 2024. 2

[12] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaussian feature splatting for real-time dynamic view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8508â8520, 2024. 2

[13] Luma AI. Interactive scenes. https://lumalabs.ai/ interactive-scenes, 2024. 1, 2

[14] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: representing scenes as neural radiance fields for view synthesis. Commun. ACM, 65(1):99â106, 2021. 1

[15] Thomas Muller, Alex Evans, Christoph Schied, and Alexan- Â¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph., 41(4), 2022. 1

[16] Omar Cornut. Dear imgui. 4

[17] PlayCanvas. PlayCanvas Engine: Powerful web graphics runtime built on webgl, webgpu, webxr and gltf. https: //github.com/playcanvas/engine, 2024. 2

[18] PlayCanvas. SuperSplat: 3d gaussian splat editor. https: //github.com/playcanvas/supersplat, 2024. 1, 2

[19] Aras Pranckevicius.Ë Unity Gaussian Splatting: A Unity implementation of 3D Gaussian Splatting. https : / / github . com / aras - p/UnityGaussianSplatting, 2023. 1, 2

[20] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10318â10327, 2021. 1

[21] Rong Liu. Gaussian splatting monitor. 2

[22] Scaniverse. Scaniverse: 3d scanning, Gaussian Splatting and LiDAR. https://scaniverse.com, 2024. 2

[23] sparkjsdev. spark: An advanced 3d gaussian splatting renderer for three.js. https://github.com/ sparkjsdev/spark, 2024. 2

[24] Spline. Spline: Design tool for 3D web experiences. https://spline.design/3d-design. 1

[25] Matthew Tancik, Ethan Weber, Evonne Ng, Ruilong Li, Brent Yi, Justin Kerr, Terrance Wang, Alexander Kristoffersen, Jake Austin, Kamyar Salahi, Abhik Ahuja, David McAllister, and Angjoo Kanazawa. Nerfstudio: A modular framework for neural radiance field development. In ACM SIGGRAPH 2023 Conference Proceedings, 2023. 2

[26] Unity Technologies. Native Rendering Plugin example for Unity. https : / / github . com / Unity - Technologies/NativeRenderingPlugin, 2024. 3

[27] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20310â 20320, 2024. 2

[28] Zhen Xu, Tao Xie, Sida Peng, Haotong Lin, Qing Shuai, Zhiyuan Yu, Guangzhao He, Jiaming Sun, Hujun Bao, and Xiaowei Zhou. Easyvolcap: Accelerating neural volumetric video research. 2023. 2

[29] Zhen Xu, Sida Peng, Haotong Lin, Guangzhao He, Jiaming Sun, Yujun Shen, Hujun Bao, and Xiaowei Zhou. 4k4d: Real-time 4d view synthesis at 4k resolution. In CVPR, 2024. 2

[30] Jinbo Yan, Rui Peng, Luyang Tang, and Ronggang Wang. 4d gaussian splatting with scale-aware residual field and adaptive optimization for real-time rendering of temporally complex dynamic scenes. In ACM Multimedia 2024. 2

[31] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. arXiv preprint arXiv:2309.13101, 2023. 2

[32] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. Realtime photorealistic dynamic scene representation and rendering with 4d gaussian splatting. In International Conference on Learning Representations (ICLR), 2024. 2

[33] Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey

Hu, Matthew Tancik, and Angjoo Kanazawa. gsplat: An open-source library for gaussian splatting. Journal of Machine Learning Research, 26(34):1â17, 2025. 2

[34] Brent Yi, Chung Min Kim, Justin Kerr, Gina Wu, Rebecca Feng, Anthony Zhang, Jonas Kulhanek, Hongsuk Choi, Yi Ma, Matthew Tancik, and Angjoo Kanazawa. Viser: Imperative, web-based 3d visualization in python. arXiv preprint arXiv:2507.22885, 2025. 2

[35] Youyi Zhan, Tianjia Shao, Yin Yang, and Kun Zhou. Realtime High-fidelity Gaussian Human Avatars with Positionbased Interpolation of Spatially Distributed MLPs . In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26297â26307, Los Alamitos, CA, USA, 2025. IEEE Computer Society. 3, 4