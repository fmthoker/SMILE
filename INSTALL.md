# SMILE Installation

This project relies on several open-source libraries. We recommend using **`conda`** to manage your Python environment and installing dependencies via the provided `environment.yml` file.

## Installation Steps
1. **Clone the repository**
```bash
git clone https://github.com/fmthoker/SMILE.git
cd SMILE
```
2. **Create a conda environment**
```bash
conda env create -f environment.yml
```
3. **Activate the environment**
```bash
conda activate smile
```
4. **Download CLIP weights (Optional)**
```bash
mkdir clip_weights
```
If you plan to pretrain from scratch, please download the [CLIP weights](https://huggingface.co/fmthoker/SMILE/resolve/main/clip_weights/ViT-B-16.pt) and place them in the `clip_weights` folder created above.

