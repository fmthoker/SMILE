# Official PyTorch Implementation of SMILE (CVPR 2025).

![SMILE Framework](figs/videomae.jpg)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)<br>
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/fmthoker/SMILE)


> [**SMILE: Infusing Spatial and Motion Semantics in Masked Video Learning**](https://arxiv.org/abs/2504.00527)<br>
> [Fida Mohammad Thoker](https://fmthoker.github.io/), [Letian Jiang](https://tonnew5418.github.io/), [Chen Zhao](https://zhao-chen.com/), [Bernard Ghanem](https://cemse.kaust.edu.sa/profiles/bernard-ghanem)<br>King Abdullah University of Science and Technology (KAUST)

## 📰 News
<!-- **[2022.4.24]**  Code and pre-trained models are available now! <br> -->
**[2025.5.28]** Code and pre-trained models will be released here. Welcome to **watch** this repository for the latest updates.

## ✨ Highlights

### 🔥 Masked Video Modeling for Video Pre-Training

VideoMAE performs the task of masked video modeling for video pre-training. We propose the **extremely high** masking ratio (90%-95%) and **tube masking** strategy to create a challenging task for self-supervised video pre-training.

### ⚡️ A Simple, Efficient and Strong Baseline in SSVP

VideoMAE uses the simple masked autoencoder and **plain ViT** backbone to perform video self-supervised learning. Due to the extremely high masking ratio, the pre-training time of VideoMAE is **much shorter** than contrastive learning methods (**3.2x** speedup). VideoMAE can serve as **a simple but strong baseline** for future research in self-supervised video pre-training.

### 😮 High performance, but NO extra data required

VideoMAE works well for video datasets of different scales and can achieve **87.4%** on Kinects-400, **75.4%** on Something-Something V2, **91.3%** on UCF101, and **62.6%** on HMDB51. To our best knowledge, VideoMAE is the **first** to achieve the state-of-the-art performance on these four popular benchmarks with the **vanilla ViT** backbones while **doesn't need** any extra data or pre-trained models.

## 🚀 Main Results

### ✨ Something-Something V2

|  Method  | Extra Data | Backbone | Resolution | #Frames x Clips x Crops | Top-1 | Top-5 |
| :------: | :--------: | :------: | :--------: | :---------------------: | :---: | :---: |
| VideoMAE |  ***no***  |  ViT-S   |  224x224   |         16x2x3          | 66.8  | 90.3  |
| VideoMAE |  ***no***  |  ViT-B   |  224x224   |         16x2x3          | 70.8  | 92.4  |
| VideoMAE |  ***no***  |  ViT-L   |  224x224   |         16x2x3          | 74.3  | 94.6  |
| VideoMAE |  ***no***  |  ViT-L   |  224x224   |         32x1x3          | 75.4  | 95.2  |

### ✨ Kinetics-400

|  Method  | Extra Data | Backbone | Resolution | #Frames x Clips x Crops | Top-1 | Top-5 |
| :------: | :--------: | :------: | :--------: | :---------------------: | :---: | :---: |
| VideoMAE |  ***no***  |  ViT-S   |  224x224   |         16x5x3          | 79.0  | 93.8  |
| VideoMAE |  ***no***  |  ViT-B   |  224x224   |         16x5x3          | 81.5  | 95.1  |
| VideoMAE |  ***no***  |  ViT-L   |  224x224   |         16x5x3          | 85.2  | 96.8  |
| VideoMAE |  ***no***  |  ViT-H   |  224x224   |         16x5x3          | 86.6  | 97.1  |
| VideoMAE |  ***no***  |  ViT-L   |  320x320   |         32x4x3          | 86.1  | 97.3  |
| VideoMAE |  ***no***  |  ViT-H   |  320x320   |         32x4x3          | 87.4  | 97.6  |

### ✨ AVA 2.2

Please check the code and checkpoints in [VideoMAE-Action-Detection](https://github.com/MCG-NJU/VideoMAE-Action-Detection).
|  Method  |  Extra Data  | Extra Label | Backbone | #Frame x Sample Rate | mAP  |
| :------: | :----------: | :---------: | :------: | :------------------: | :--: |
| VideoMAE | Kinetics-400 |   &cross;   |  ViT-S   |         16x4         | 22.5 |
| VideoMAE | Kinetics-400 |   &check;   |  ViT-S   |         16x4         | 28.4 |
| VideoMAE | Kinetics-400 |   &cross;   |  ViT-B   |         16x4         | 26.7 |
| VideoMAE | Kinetics-400 |   &check;   |  ViT-B   |         16x4         | 31.8 |
| VideoMAE | Kinetics-400 |   &cross;   |  ViT-L   |         16x4         | 34.3 |
| VideoMAE | Kinetics-400 |   &check;   |  ViT-L   |         16x4         | 37.0 |
| VideoMAE | Kinetics-400 |   &cross;   |  ViT-H   |         16x4         | 36.5 |
| VideoMAE | Kinetics-400 |   &check;   |  ViT-H   |         16x4         | 39.5 |
| VideoMAE | Kinetics-700 |   &cross;   |  ViT-L   |         16x4         | 36.1 |
| VideoMAE | Kinetics-700 |   &check;   |  ViT-L   |         16x4         | 39.3 |

### ✨ UCF101 & HMDB51

|  Method  |  Extra Data  | Backbone | UCF101 | HMDB51 |
| :------: | :----------: | :------: | :----: | :----: |
| VideoMAE |   ***no***   |  ViT-B   |  91.3  |  62.6  |
| VideoMAE | Kinetics-400 |  ViT-B   |  96.1  |  73.3  |

## 🔨 Installation

Please follow the instructions in [INSTALL.md](INSTALL.md).

## ➡️ Data Preparation

Please follow the instructions in [DATASET.md](DATASET.md) for data preparation.

## 🔄 Pre-training

The pre-training instruction is in [PRETRAIN.md](PRETRAIN.md).

## ⤴️ Fine-tuning with pre-trained models

The fine-tuning instruction is in [FINETUNE.md](FINETUNE.md).

## 📍Model Zoo

We provide pre-trained and fine-tuned models in [MODEL_ZOO.md](MODEL_ZOO.md).

## 👀 Visualization

We provide the script for visualization in [`vis.sh`](vis.sh).  Colab notebook for better visualization is coming soon.

## ☎️ Contact 

Zhan Tong: tongzhan@smail.nju.edu.cn

## 👍 Acknowledgements

We sincerely thank [Michael Dorkenwald](https://mdorkenwald.com/) for providing the object image dataset that supports this work.<br>
This project is built upon [VideoMAE](https://github.com/MCG-NJU/VideoMAE) and [tubelet-contrast](https://github.com/fmthoker/tubelet-contrast). Thanks to the contributors of these great codebases.

## 🔒 License

The majority of this project is released under the CC-BY-NC 4.0 license as found in the [LICENSE](https://github.com/MCG-NJU/VideoMAE/blob/main/LICENSE) file. Portions of the project are available under separate license terms: [SlowFast](https://github.com/facebookresearch/SlowFast) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) are licensed under the Apache 2.0 license. [BEiT](https://github.com/microsoft/unilm/tree/master/beit) is licensed under the MIT license.

## ✏️ Citation

If you think this project is helpful, please feel free to leave a star⭐️ and cite our paper:

```
@inproceedings{tong2022videomae,
  title={Video{MAE}: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Zhan Tong and Yibing Song and Jue Wang and Limin Wang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

@article{videomae,
  title={VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Tong, Zhan and Song, Yibing and Wang, Jue and Wang, Limin},
  journal={arXiv preprint arXiv:2203.12602},
  year={2022}
}
```
