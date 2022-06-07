# Revisiting Image Pyramid Structure for High Resolution Salient Object Detection (InSPyReNet)

PyTorch implementation of Revisiting Image Pyramid Structure for High Resolution Salient Object Detection (InSPyReNet)

## Abstract

  Salient object detection (SOD) has been in the spotlight recently, yet has been studied less for high-resolution (HR) images. 
  Unfortunately, HR images and their pixel-level annotations are certainly more labor-intensive and time-consuming compared to low-resolution (LR) images.
  Therefore, we propose an image pyramid-based SOD framework, Inverse Saliency Pyramid Reconstruction Network (InSPyReNet), for HR prediction without any of HR datasets.
  We design InSPyReNet to produce a strict image pyramid structure of saliency map, which enables to ensemble multiple results with pyramid-based image blending.
  For HR prediction, we design a pyramid blending method which synthesizes two different image pyramids from a pair of LR and HR scale from the same image to overcome effective receptive field (ERF) discrepancy. Our extensive evaluation on public LR and HR SOD benchmarks demonstrates that InSPyReNet surpasses the State-of-the-Art (SotA) methods on various SOD metrics and boundary accuracy.

## Architecture

InSPyReNet                 |  pyramid blending
:-------------------------:|:-------------------------:
![](./figures/fig_architecture.png)  |  ![](./figures/fig_pyramid_blending.png)

## 1. Create environment
  + Create conda environment with following command `conda create -y -n inspyrenet python=3.8`
  + Activate environment with following command `conda activate inspyrenet`
  + Install requirements with following command `pip install -r requirements.txt`
  
## 2. Preparation
  * [Dataset](https://drive.google.com/file/d/1Aft2Wm0-NmvZ-ezZH-DfHM30OBc-ZOIi/view?usp=sharing) --> `data/RGB_Dataset/Train_Dataset/...`
  * [Res2Net50 checkpoint](https://drive.google.com/file/d/1MMhioAsZ-oYa5FpnTi22XBGh5HkjLX3y/view?usp=sharing), [SwinB checkpoint](https://drive.google.com/file/d/1fBJFMupe5pV-Vtou-k8LTvHclWs0y1bI/view?usp=sharing) --> `data/backbone_ckpt/...`
  * Train with extra training datasets (HRSOD, UHRSD):
  ```
  Train:
    Dataset:
        type: "RGB_Dataset"
        root: "data/RGB_Dataset/Train_Dataset"
        sets: ['DUTS-TR'] --> ['DUTS-TR', 'HRSOD-TR-LR', 'UHRSD-TR-LR']
  ```

## 3. Train & Evaluate
  * Train InSPyReNet (SwinB)
  ```
  python run/Train.py --config configs/InSPyReNet_SwinB.yaml --verbose
  ```
  * Inference for test benchmarks
  ```
  python run/Test.py --config configs/InSPyReNet_SwinB.yaml --verbose
  ```
  * Evaluate metrics
  ```
  python run/Eval.py --config configs/InSPyReNet_SwinB.yaml --verbose
  ```

## 4. Checkpoints

Model                      |  Train DB                          
:-|:-
[InSPyReNet (Res2Net50)](https://drive.google.com/file/d/12moRuU8F0-xRvE16bVg6mkGWDuqYHJor/view?usp=sharing) | DUTS-TR                             
[InSPyReNet (SwinB)](https://drive.google.com/file/d/1k5hNJImgEgSmz-ZeJEEb_dVkrOnswVMq/view?usp=sharing)         | DUTS-TR                             
[InSPyReNet (SwinB)](https://drive.google.com/file/d/1nbs6Xa7NMtcikeHFtkQRVrsHbBRHtIqC/view?usp=sharing)         | DUTS-TR, HRSOD-TR-LR                
[InSPyReNet (SwinB)](https://drive.google.com/file/d/1uLSIYXlRsZv4Ho0C-c87xKPhmF_b-Ll4/view?usp=sharing)         | HRSOD-TR-LR, UHRSD-TR-LR            
[InSPyReNet (SwinB)](https://drive.google.com/file/d/14gRNwR7XwJ5oEcR4RWIVbYH3HEV6uBUq/view?usp=sharing)         | DUTS-TR, HRSOD-TR-LR, UHRSD-TR-LR

* LR denotes resized into low-resolution scale (i.e. 384 x 384) since we do not need HR datasets.
  
## 5. Citation

+ Backbones:
  + Res2Net: [A New Multi-scale Backbone Architecture](https://github.com/Res2Net/Res2Net-PretrainedModels)
  + Swin Transformer: [Hierarchical Vision Transformer using Shifted Windows](https://github.com/microsoft/Swin-Transformer)
+ Datasets:
  + [DUTS](http://saliencydetection.net/duts/)
  + [DUT-OMRON](http://saliencydetection.net/dut-omron/)
  + [ECSSD](https://i.cs.hku.hk/~gbli/deep_saliency.html)
  + [HKU-IS](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
  + [PASCAL-S](http://cbi.gatech.edu/salobj/)

+ Evaluation Toolkit: [PySOD Metrics](https://github.com/lartpang/PySODMetrics)
