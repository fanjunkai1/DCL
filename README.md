<p align=center><img src="doc/logo.jpg" width="400px", height="90px"> </p>

<div align="center">
    <h1>Depth-Centric Dehazing and Depth-Estimation from Real-World Hazy Driving Video (AAAI 2025)</h1>
    <div>
        <a href='https://fanjunkai1.github.io/' target='_blank'>Junkai Fan</a><sup>1</sup>&emsp;
        <a href='https://w2kun.github.io/' target='_blank'>Kun Wang</a><sup>1</sup>&emsp;
        <a href='https://yanzq95.github.io/' target='_blank'>Zhiqiang Yan</a><sup>1</sup>&emsp;
        <a href='https://cschenxiang.github.io/' target='_blank'>Xiang Chen</a><sup>1</sup>&emsp;
        <a target='_blank'>Shangbing Gao</a><sup>2</sup>&emsp;
        <a href='https://sites.google.com/view/junlineu/' target='_blank'>Jun Li</a><sup>1*</sup>&emsp;
        <a href='https://scholar.google.com/citations?user=6CIDtZQAAAAJ&hl=zh-CN' target='_blank'>Jian Yang</a><sup>1*</sup>
    </div>
    <div>
        <sup>1</sup>PCA Lab, Nanjing University of Science and Technology<br><sup>2</sup>Huaiyin Institute of Technology
    </div>
    <div>
        <h2 align="center">
            <a href="https://fanjunkai1.github.io/projectpage/DCL/chinese_interpretation.html" target='_blank'>[中文解读]</a>
        </h2>
    </div>
</div>

<div align="center">
    
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/pdf/2412.11395)
[![Website](doc/badge-website.svg)](https://fanjunkai1.github.io/projectpage/DCL/index.html)
[![Video](https://img.shields.io/badge/YouTube-Video-c4302b?logo=youtube&logoColor=red)](https://www.youtube.com/watch?v=8FYw-MHksq4)
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=fanjunkai1/DCL)

</div>

We propose a novel depth-centric learning (DCL) framework that integrates the atmospheric scattering model (ASM) with the brightness consistency constraint (BCC) constraint. Our key idea is that both ASM and BCC rely on a shared depth estimation network. This network simultaneously exploits adjacent dehazed frames to enhance depth estimation via BCC and uses the refined depth cues to more effectively remove haze through ASM.

<p align="center">
  <img src="doc/demo.gif" alt="example input output gif" width="600" />
</p>

For more **video demos**, please refer to our [project homepage](https://fanjunkai1.github.io/projectpage/DCL/index.html).

## 📢 News
- [16-12-2024] The "Chinese Interpretation" version of DCL has been added.
- [14-12-2024] Training and inference code is released. (this repository).
- [13-12-2024] We created the [project homepage](https://fanjunkai1.github.io/projectpage/DCL/index.html) and the GitHub README.
- [10-12-2024] Accepted to AAAI 2025.
  
## DCL Pipeline

Our Depth-Centric Learning (DCL) framework integrates the atmospheric scattering model with a brightness consistency constraint via shared depth prediction. $D_{MFIR}$ improves high-frequency detail in dehazed frames, while $D_{MDR}$ reduces black holes in depth maps from weakly textured areas.

<p align="center">
  <img src="doc/pipeline.png" width="750" height='500' />
</p>

## ⚙️ Dependencies and Installation

### Initialize Conda Environment and Clone Repo
```bash
git clone https://github.com/fanjunkai1/DCL.git

conda create -n DCL python=3.9.19
conda activate DCL

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

```

### Intall Dependencies

```bash
pip install -r requirements.txt
```

## Download and Preprocess dataset

The original DVD (CVPR 2024) dataset (1920x1080 size) can be downloaded from the following link:

- **GoProHazy** dataset can be downloaded from [Baidu Drive](https://pan.baidu.com/s/1u_jFzZtUhG1528e1kkGmUQ#list/path=%2F)(hbih).
- **DrivingHazy** dataset can be downloaded from [Baidu Drive](https://pan.baidu.com/s/1gQTV6F9bwnmKtmUohzi1Nw#list/path=%2F)(ei4j).
- **InternetHazy** dataset can be downloaded from [Baidu Drive](https://pan.baidu.com/s/1WIZNwFH-re8ty6zJPjct6g#list/path=%2F)(p39a).

Google Drive users can download the GoProHazy, DrivingHazy, and InternetHazy datasets via this [link](https://drive.google.com/drive/folders/11CmFXT32a3QkCXc76-J_Wx2kgpE_hALu?dmr=1&ec=wgc-drive-globalnav-goto)


## 🚀 Preprocess dataset

All videos in these datasets are initially recorded at a resolution of 1920×1080. After applying distortion correction and cropping based on the intrinsic parameters K of the GoPro 11 camera (calibrated by us), the resolutions of GoProHazy and DrivingHazy are 1600×512.

1. **Camera Calibration**. Download the chessboard image set from [here](https://drive.google.com/drive/folders/12b_XvtUs7oc9HjfoS1JepmUJC2eI18vm), and place the data in the `./calibrate folder`, then run the following command:
   
```bash
python calibrate.py
```
2. **Undistort and Crop**. Use the calibration results to undistort and crop the 1920x1080 image to 1600x512, then save the new intrinsic parameters. The command is as follows:
   
```bash
python preprocess.py
```

The data organization for the `./preprocess` folder is shown below:

~~~
preprocess
|--gopro_preprocess
   |--videos
      |--test
         |--clear_video
            |--...
         |--hazy_video
            |--...
         |--processed
            |--...
      |-- train
         |--clear_video
            |--1_clear_video.mp4
            |--...
         |--hazy_video
            |--1_hazy_video.mp4
            |--...
         |--processed
            |--clear_video
            |--hazy_video
            |--intrinsic.npy
~~~

**Note**: The preprocessed GoProHazy dataset can be downloaded [here](https://drive.google.com/drive/folders/12b_XvtUs7oc9HjfoS1JepmUJC2eI18vm).

## 🏃 Data Split

The data format in the `./split/gopro_fan` folder and the corresponding explanation are as follows:
```bash
......
train_video/hazy_video/7_hazy_video 6 5 4 5
train_video/hazy_video/7_hazy_video 7 5 5 6
train_video/hazy_video/7_hazy_video 8 6 5 7
......
```

|    the folder of current frame t  |t idx|t matched idx|t-1 matched idx|t+1 matched idx|
|:---------------------------------:|:---:|:-----------:|:-------------:|:-------------:|
|train_video/hazy_video/7_hazy_video|  7  |       5     |       5       |       6       |


## 🏋️ Training DCL

1. **Training model**. Place the *gopro_data* (preprocessed GoProHazy) folder, downloaded from [Google Drive](https://drive.google.com/drive/folders/12b_XvtUs7oc9HjfoS1JepmUJC2eI18vm), into the `./data` folder, and then execute the following command:

```bash
python train.py --model_name DCL
```

2. **Training Visualization**. The training and validation log files for DCL are saved in the train and val folders under `./logger/DCL`. They can be visualized in TensorBoard using the following command:

```bash
cd DCL
tensorboard --logdir=./logger/DCL
```
To visualize locally from a remote server, run `ssh -L 16006:127.0.0.1:6006 -p xxxxx root@serverIP` to set up the tunnel, then use the command above. Finally, open http://localhost:16006/ in your browser for training visualization.

## ⚡ Inference DCL on GoProHazy

Move the trained model from `./logger/DCL/models` to the `./models/DCL folder`, or use our pre-trained model, which can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1sNgOyh-DIuxG3sdty_D24Ugo35qJ7dlr). Then, execute the following command:

```bash
python test_gopro_hazy.py --image_path ./data/gopro_data/test_video/hazy_video/24_hazy_video
```

## 🎮 Inference DCL on DENSE-Fog

```bash
python test_dense_fog.py --image_path /opt/data/common/SeeingThroughFog/SeeingThroughFogCompressedExtracted --dataset densefog --load_weights_folder ./models/DCL
python test_dense_fog.py --image_path /opt/data/common/SeeingThroughFog/SeeingThroughFogCompressedExtracted --dataset lightfog --load_weights_folder ./models/DCL
```
The output test results are saved in the `./outputs` folder.


## 🔍 Results

Our DCL achieved state-of-the-art performance on *GoProHazy* and *DENSE-Fog* datasets,

<details open> 
<summary>Visual Comparison (click to expand)</summary>

- Visual comparison for video dehazing
  <p align="center">
  <img width="750" src="doc/video_dehazing-results.png">
  </p>
- Visual comparison for depth estimation
  <p align="center">
  <img width="750" src="doc/depth_estimation-results.png">
  </p>
  </p>
  
  </details>

## 🎓 Citation

If you find the code helpful in your research or work, please cite the following paper(s).

```bibtex
@inproceedings{fan2025depth,
  title={Depth-Centric Dehazing and Depth-Estimation from Real-World Hazy Driving Video},
  author={Fan, Junkai and Wang, Kun and Yan, Zhiqiang and Chen, Xiang and Gao, Shangbing and Li, Jun and Yang, Jian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  pages={2852--2860},
  year={2025}
}

@inproceedings{fan2024driving,
  title={Driving-Video Dehazing with Non-Aligned Regularization for Safety Assistance},
  author={Fan, Junkai and Weng, Jiangwei and Wang, Kun and Yang, Yijun and Qian, Jianjun and Li, Jun and Yang, Jian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={26109--26119},
  year={2024}
}
```

## 🤗 Acknowledgment
This code is based on the [Monodepth2](https://github.com/nianticlabs/monodepth2) and [DVD](https://github.com/fanjunkai1/DVD). Thank them for their outstanding work.

## 📧 Contact
If you have any questions or suggestions, please contact junkai.fan@njust.edu.cn

## 🎫 License

This work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

By downloading and using the code and model you agree to the terms in the  [LICENSE](LICENSE.txt).

[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)




