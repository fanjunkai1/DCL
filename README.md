<div align="center">

<h1>Depth-Centric Dehazing and Depth-Estimation from Real-World Hazy Driving Video</h1>

<div>
    <a href='https://fanjunkai1.github.io/' target='_blank'>Junkai Fan</a><sup>1</sup>&emsp;
    <a href='https://github.com/w2kun' target='_blank'>Kun Wang</a><sup>1</sup>&emsp;
    <a href='https://yanzq95.github.io/' target='_blank'>Zhiqiang Yan</a><sup>1</sup>&emsp;
    <a href='https://cschenxiang.github.io/' target='_blank'>Xiang Chen</a><sup>1</sup>&emsp;
    <a target='_blank'>Shangbing Gao</a><sup>2</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=iGPEwQsAAAAJ&hl=zh-CN' target='_blank'>Jun Li</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=6CIDtZQAAAAJ&hl=zh-CN' target='_blank'>Jian Yang</a><sup>1</sup>
</div>

<div>
    <sup>1</sup>PCA Lab, Nanjing University of Science and Technology<br><sup>2</sup>Huaiyin Institute of Technology
</div>

<div>
    <h4 align="center">
        <a href="" target='_blank'>AAAI 2025</a>
    </h4>
</div>
</div>

We propose a novel depth-centric learning framework that integrates the atmospheric scattering model (ASM) with the brightness consistency constraint (BCC) constraint. Our key idea is that both ASM and BCC rely on a shared depth estimation network. This network simultaneously exploits adjacent dehazed frames to enhance depth estimation via BCC and uses the refined depth cues to more effectively remove haze through ASM.

![teaser](doc/video_frame_results.png)

## 📢 News
- [13-12-2024] We created the [project homepage](https://fanjunkai1.github.io/projectpage/DCL/index.html) and the GitHub README.

## 🎬 Video demo
To demonstrate the stability of the proposed method, we separately compared it with the latest SoTA video dehazing (e.g., MAP-Net, DVD) and monocular depth estimation methods (e.g., Mono-ViFI, Lite-Mono) on GoProHazy. 

https://github.com/user-attachments/assets/55369027-acf9-4e47-83a8-dfa2f982bdfc


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

For users who use Google Drive, you can download GoProHazy, DrivingHazy, and InternetHazy datasets using this [link](https://drive.google.com/drive/folders/11CmFXT32a3QkCXc76-J_Wx2kgpE_hALu?dmr=1&ec=wgc-drive-globalnav-goto)


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

|    the folder of current frame t  |t index|t matched index|t-1 matched index|t+1 matched index|
|-----------------------------------|-------|---------------|-----------------|-----------------|
|train_video/hazy_video/7_hazy_video|   7   |       5       |        5        |        6        |


## 🏋️ Training DCL

1. **Training model**. Place the *gopro_data* (preprocessed GoProHazy) folder, downloaded from [Google Drive](https://drive.google.com/drive/folders/12b_XvtUs7oc9HjfoS1JepmUJC2eI18vm), into the `./data` folder, and then execute the following command:

```bash
python train.py --model_name DCL
```

2. **Training Visualization**. The training and validation log files for DCL are saved in the train and val folders under ./logger/DCL. They can be visualized in TensorBoard using the following command:

```bash
cd DCL
tensorboard --logdir=./logger/DCL
```

## ⚡ Inference DCL on GoProHazy test set

































