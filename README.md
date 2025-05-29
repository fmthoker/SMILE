# Official PyTorch Implementation of SMILE (CVPR 2025).

![SMILE Framework](figs/smile.jpg)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)<br>
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/fmthoker/SMILE/tree/main/SMILE_MODELS)


> [**SMILE: Infusing Spatial and Motion Semantics in Masked Video Learning**](https://arxiv.org/abs/2504.00527)<br>
> [Fida Mohammad Thoker](https://fmthoker.github.io/), [Letian Jiang](https://tonnew5418.github.io/), [Chen Zhao](https://zhao-chen.com/), [Bernard Ghanem](https://cemse.kaust.edu.sa/profiles/bernard-ghanem)<br>King Abdullah University of Science and Technology (KAUST)

## 📰 News
<!-- **[2022.4.24]**  Code and pre-trained models are available now! <br> -->
**[2025.5.28]** Code and pre-trained models will be released here. Welcome to **watch** this repository for the latest updates.

## ✨ Highlights

### 🔥 State-of-the-art on SSv2 and K400

Our method achieves state-of-the-art performance on **SSv2** and **K400** benchmarks with a ViT-B backbone, surpassing prior self-supervised video models by up to **2.5%**, thanks to efficient *CLIP-based semantic supervision*.

### ⚡️ Leading Results Across Generalization Challenges

We evaluate our method on the [**SEVERE benchmark**](https://bpiyush.github.io/SEVERE-website/), covering domain shift, low-shot learning, fine-grained actions, and task adaptability. Our model consistently outperforms prior methods and achieves a **3.0% average gain** over strong baselines, demonstrating superior generalization in diverse video understanding tasks.

### 😮 Superior Motion Representation Without Video-Text Alignment

Compared to CLIP-based methods such as [**ViCLIP**](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid) and [**UMT**](https://github.com/OpenGVLab/unmasked_teacher), our model achieves higher accuracy on motion-sensitive datasets, particularly under *linear probing*. This indicates stronger video representations learned with less data and without relying on video-text alignment.

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

## ☎️ Contact 

Fida Mohammad Thoker: fida.thoker@kaust.edu.sa

## 👍 Acknowledgements

We sincerely thank [Michael Dorkenwald](https://mdorkenwald.com/) for providing the object image dataset that supports this work.<br>
This project is built upon [VideoMAE](https://github.com/MCG-NJU/VideoMAE) and [tubelet-contrast](https://github.com/fmthoker/tubelet-contrast). Thanks to the contributors of these great codebases.

## 🔒 License

This project is released under the MIT license. For more details, please refer to the [LICENSE](https://github.com/fmthoker/SMILE/blob/main/LICENSE) file.

## ✏️ Citation

If you think this project is helpful, please feel free to leave a star⭐️ and cite our paper:

```
@inproceedings{thoker2025smile,
  author    = {Thoker, Fida Mohammad and Jiang, Letian and Zhao, Chen and Ghanem, Bernard},
  title     = {SMILE: Infusing Spatial and Motion Semantics in Masked Video Learning},
  journal   = {CVPR},
  year      = {2025},
}
```
