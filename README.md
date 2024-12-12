# Depth-Centric Dehazing and Depth-Estimation from Real-World Hazy Driving Video (AAAI 2025)


This repository represents the official implementation of the paper titled "Depth-Centric Dehazing and Depth-Estimation from Real-World Hazy Driving Video".

[![Website](doc/badge-website.svg)](https://fanjunkai1.github.io/projectpage/DCL/index.html)

[Junkai Fan](https://fanjunkai1.github.io/),
[Kun Wang](https://github.com/w2kun),
[Zhiqiang Yan](https://yanzq95.github.io/),
[Xiang Chen](https://cschenxiang.github.io/),
[Shangbing Gao](),
[Jun Li](https://sites.google.com/view/junlineu/)
[Jian Yang](https://scholar.google.com/citations?user=6CIDtZQAAAAJ&hl=zh-CN)


PCA Lab, Nanjing University of Science and Technology

We propose a novel depth-centric learning framework that integrates the atmospheric scattering model (ASM) with the brightness consistency constraint (BCC) constraint. Our key idea is that both ASM and BCC rely on a shared depth estimation network. This network simultaneously exploits adjacent dehazed frames to enhance depth estimation via BCC and uses the refined depth cues to more effectively remove haze through ASM.

![teaser](doc/video_frame_results.png)

## 📢 News
- [13-12-2024] We created the [project homepage](https://fanjunkai1.github.io/projectpage/DCL/index.html) and the GitHub README.

## 🎬 Video demo
To demonstrate the stability of the proposed method, we separately compared it with the latest SoTA video dehazing (e.g., MAP-Net, DVD) and monocular depth estimation methods (e.g., Mono-ViFI, Lite-Mono) on GoProHazy. 

https://github.com/user-attachments/assets/55369027-acf9-4e47-83a8-dfa2f982bdfc

