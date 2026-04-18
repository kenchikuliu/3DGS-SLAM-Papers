# Capture and Interact: Rapid 3D Object Acquisition and Rendering with Gaussian Splatting in Unity

<!-- image-->  
Figure 1: System pipeline: (1) video capture and upload to a cloud server; (2) video decomposition into individual frames; (3) SfM application to generate a sparse point cloud; (4) training of a 3D GS model on this point cloud; (5) download of the resulting Gaussian representation to a local device and interactive rendering via Unity.

## Abstract

Capturing and rendering three-dimensional (3D) objects in real time remain a significant challenge, yet hold substantial potential for applications in augmented reality, digital twin systems, remote collaboration and prototyping. We present an end-to-end pipeline that leverages 3D Gaussian Splatting (3D GS) to enable rapid acquisition and interactive rendering of real-world objects using a mobile device, cloud processing and a local computer. Users scan an object with a smartphone video, upload it for automated 3D reconstruction, and visualize it interactively in Unity at an average of 150 frames per second (fps) on a laptop. The system integrates mobile capture, cloud-based 3D GS and Unity rendering to support real-time telepresence. Our experiments show that the pipeline processes scans in approximately 10 minutes on a graphics processing unit (GPU) achieving real-time rendering on the laptop.

## CCS Concepts

â¢ Applied computing â Representation;

## Keywords

3D reconstruction, 3D Gaussian splatting, interactive telepresence, computer vision, machine learning, real-time rendering

## 1 Introduction

Real-time 3D object capture and rendering have a broad applications in education, virtual collaboration, healthcare and extended reality (XR) [1]. Traditional pipelines rely on expensive hardware, controlled environments and post-processing, limiting accessibility for interactive or mobile scenarios.

Point clouds, voxels and triangular meshes [9] all have drawbacks: sparsity, memory intensity, high reconstruction cost and dependence on expensive hardware, such as Light Detection and Ranging (LiDAR), with high-quality input data [10]. Yet, they still lack in the visual realism needed to accurately capture real-world objects.

Neural Radiance Fields (NeRF) [5] achieve photorealistic rendering by modeling scenes as continuous functions over 3D coordinates and view directions. However, their long training times and computational intensity make them unsuitable for real-time or mobile applications such as augmented reality and live telepresence. MobileNeRF addresses this by enabling mobile rendering of pre-trained scenes using a compact rasterization-based approach, but it lacks on-device capture and end-to-end reconstruction [2].

Karb et al. [4] introduce 3D Gaussian Splatting (3D GS), which models the scene as a set of view-dependent Gaussians that are directly splatted onto the image plane using a fast, differentiable rasterization process. This approach enables faster training and real-time rendering performance while maintaining photorealistic quality, making it well-suited for interactive applications and realtime 3D content generation on consumer hardware.

We propose a cloud-based 3D GS pipeline that reconstructs and renders objects captured via mobile videos. The result is an interactive, photorealistic visualization, rendered in Unity on a client device. Our system demonstrates a practical approach to democratizing 3D content creation.

## 2 System Overview

The process begins with a short video recorded through a web interface on any mobile device. Once uploaded to a cloud server, the script decomposes the video into frames and processes using Structure-from-Motion (SfM) to estimate camera poses and generate a sparse point cloud.

<!-- image-->  
Figure 2: Web interface.

During training, the algorithm converts each point to a 3D Gaussian, an elliptical blob defined by position, scale, rotation, opacity, color and spherical harmonic coefficients. The system applies viewdependent shading and optimizes the Gaussians over approximately 30K iterations, actively adjusting, splitting, or pruning them based on their contribution to photometric reconstruction accuracy. This results in a detailed and photorealistic 3D representation.

After optimization, the client device downloads the resulting Gaussian model and renders interactively in Unity using a real-time GPU splatting renderer. Figure 1 illustrates the full pipeline, from capture to interactive rendering.

## 3 Implementation and Evaluation

The system consists of three components: mobile video capture, cloud-based 3D GS and Unity-based real-time rendering. The mobile front-end, built with HTML5 and JavaScript, allows users to record and upload a short video (see Figure 2). On the server side, the Structure-from-Motion in COLMAP [7, 8] extracts camera poses and reconstructs a sparse point cloud. The 3D GS pipeline relies on the open-source implementation by Kerbl et al. [3].

We evaluate the pipeline using both simple objects (e.g., cups and books) and complex ones (e.g., flowers, sunglasses) objects. The optimization runs for approximately 30K iterations on an NVIDIA RTX A40 GPU and the whole pipeline completes in an average of 10 minutes depending on scene complexity and video duration. On average, the reconstructed objects achieve a Peak Signal-to-Noise Ratio (PSNR) of 34.65. In future work, we plan to explore acceleration techniques for time-sensitive applications.

After the client receives the resulting set of Gaussians for local rendering, a Unity-based renderer [6] enables real-time interaction, including pan, zoom, rotation and scaling (Figure 3). The renderer operates at an average of 150 fps for approximately 500K splats on a MacBook Pro with the M4 chip.

## 4 Demo Setup

The demonstration involves a mobile device and a laptop, both connected to a cloud server. Attendees record a short video using the mobile interface. The system processes the video in the cloud and renders the reconstructed model in Unity within 10 minutes. Users then interact with the result on the local laptop. To ensure a stable and seamless demonstration, we also provide pre-captured demo objects rendered live in Unity.

## 5 Conclusion and Future Work

Our system enables rapid 3D object capture and rendering using only mobile devices, unlocking a wide range of applications across academic, educational and creative domains. Its modular and accessible design supports real-time 3D visualization. Future optimizations, such as using transformer-based feature generation instead of SfM, promise to reduce latency for time-critical use cases like live events.

The system serves diverse domains by enabling easy 3D digitization and interaction. Museums, schools and similar institutions have the potential to use it to digitize physical artifacts for documentation, virtual exhibits and hands-on learning. Researchers and teachers without 3D modeling skills create high-quality models for analysis or education. In industrial design and prototyping, users scan mockups or parts, integrate them into digital workflows and interact with them in real time, speeding up the design process.

By lowering the barrier to high-quality 3D content creation, the system encourages broader adoption of 3D GS for practical use. Future enhancements include increasing the interactivity of the viewer through integration with large language models (LLMs). For example, users gain the ability to manipulate the 3D scene intuitively using voice or text-based commands, such as "Change the color of the cup to yellow." This natural language interface enables a more accessible and hands-free experience, which proves especially valuable in educational, collaborative and assistive environments.

<!-- image-->

<!-- image-->  
Figure 3: Examples of rendered objects.

## References

[1] Samia Allaoua Chelloug, Hamid Ashfaq, Suliman A Alsuhibany, Mohammad Shorfuzzaman, Abdulmajeed Alsufyani, Ahmad Jalal, et al. 2023. Real Objects Understanding Using 3D Haptic Virtual Reality for E-Learning Education. Computers, Materials & Continua 75, 1 (2023).

[2] Zhiqin Chen, Thomas Funkhouser, Peter Hedman, and Andrea Tagliasacchi. 2023. Mobilenerf: Exploiting the Polygon Rasterization Pipeline for Efficient Neural Field Rendering on Mobile Architectures. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 16569â16578.

[3] INRIA. 2023. GitHub - 3D Gaussian Splatting for Real-Time Radiance Field Rendering â github.com. https://github.com/graphdeco-inria/gaussian-splatting. [Accessed 29-05-2025].

[4] Bernhard Kerbl, Georgios Kopanas, Thomas LeimkÃ¼hler, and George Drettakis. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics 42, 4 (July 2023). https://repo-sam.inria.fr/fungraph/3dgaussian-splatting/

[5] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2021. Nerf: Representing Scenes as Neural Radiance Fields for View synthesis. Commun. ACM 65, 1 (2021), 99â106.

[6] Aras PranckeviÄius. 2023. GitHub - aras-p/UnityGaussianSplatting: Toy Gaussian Splatting visualization in Unity â github.com. https://github.com/arasp/UnityGaussianSplatting. [Accessed 27-05-2025].

[7] Johannes Lutz SchÃ¶nberger and Jan-Michael Frahm. 2016. Structure-from-Motion Revisited. In Conference on Computer Vision and Pattern Recognition (CVPR).

[8] Johannes Lutz SchÃ¶nberger, Enliang Zheng, Marc Pollefeys, and Jan-Michael Frahm. 2016. Pixelwise View Selection for Unstructured Multi-View Stereo. In European Conference on Computer Vision (ECCV).

[9] Jeroen Van Der Hooft, Hadi Amirpour, Maria Torres Vega, Yago Sanchez, Raimund Schatz, Thomas Schierl, and Christian Timmerer. 2023. A Tutorial on Immersive Video Delivery: From Omnidirectional Video to Holography. IEEE Communications Surveys & Tutorials 25, 2 (2023), 1336â1375.

[10] Qian Wang, Yi Tan, and Zhongya Mei. 2020. Computational Methods of Acquisition and Processing of 3D Point Cloud Data for Construction Applications. Archives of computational methods in engineering 27, 2 (2020), 479â499.