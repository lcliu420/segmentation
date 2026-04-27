# MADGNet
This repository is trimmed to the ISIC2018 dermoscopy experiment of MADGNet (CVPR 2024).

## Abstract
Generalizability in deep neural networks plays a pivotal role in medical image segmentation. However, deep learning-based medical image analyses tend to overlook the importance of frequency variance, which is critical element for achieving a model that is both modality-agnostic and domain-generalizable.  Additionally, various models fail to account for the potential information loss that can arise from multi-task learning under deep supervision, a factor that can impair the model’s representation ability. To address these challenges, we propose a Modality-agnostic Domain Generalizable Network (MADGNet) for medical image segmentation, which comprises two key components: a Multi-Frequency in Multi-Scale Attention (MFMSA) block and Ensemble Sub-Decoding Module (E-SDM). The MFMSA block refines the process of spatial feature extraction, particularly in capturing boundary features, by incorporating multi-frequency and multi-scale features, thereby offering informative cues for tissue outline and anatomical structures. Moreover, we propose E-SDM to mitigate information loss in multi-task learning with deep supervision, especially during substantial upsampling from low resolution. We evaluate the segmentation performance of MADGNet across six modalities and fifteen datasets. Through extensive experiments, we demonstrate that MADGNet consistently outperforms  state-of-the-art models across various modalities, showcasing superior segmentation performance.  This affirms MADGNet as a robust solution for medical image segmentation that excels in diverse imaging scenarios. 

## Overall Architecture of MADGNet
![MFMSNet](https://github.com/BlindReview922/MADGNet/assets/142275582/8c1d54c5-b03d-4c71-b7f1-81e8c91e0d36)
![CascadedDecoder](https://github.com/BlindReview922/MADGNet/assets/142275582/8c057fd3-e681-4b52-b630-591f4bc5a8f5)

## ISIC2018 Only Workflow

1. Place the ISIC2018 images and masks under:

```
dataset/BioMedicalDataset/ISIC2018/
├── ISIC2018_Task1-2_Training_Input/
├── ISIC2018_Task1_Training_GroundTruth/
├── train_frame.csv
├── val_frame.csv
└── test_frame.csv
```

2. Install dependencies:

```bash
pip install torch torchvision numpy pandas pillow scipy scikit-learn scikit-image
```

3. Train on ISIC2018:

```bash
python IS2D_main.py --train --epochs 100 --batch_size 16 --num_workers 4
```

4. Evaluate the best checkpoint on the test split:

```bash
python IS2D_main.py --eval_split test
```

5. Optionally save binary prediction masks:

```bash
python IS2D_main.py --eval_split test --save_predictions
```

The training setup follows the paper for dermoscopy:
- `352 x 352` input size
- Adam with initial learning rate `1e-4`
- cosine annealing to `1e-6`
- batch size `16`
- `100` epochs for dermoscopy
- horizontal/vertical flips and `[-5°, 5°]` rotation
- deep supervision on region, distance map, and boundary tasks

# Bibtex
```
@InProceedings{Nam_2024_CVPR,
    author    = {Nam, Ju-Hyeon and Syazwany, Nur Suriza and Kim, Su Jung and Lee, Sang-Chul},
    title     = {Modality-agnostic Domain Generalizable Medical Image Segmentation by Multi-Frequency in Multi-Scale Attention},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {11480-11491}
}
```
