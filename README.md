# Contactless Human Activity Recognition using Deep Learning

This repository contains the implementation of **Contactless Human Activity Recognition (HAR)** using Channel State Information (CSI) and deep learning models. The project explores the transformation of raw CSI signals into pseudocolor images and evaluates the performance of both **2D Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)**.

---

## Overview
- **Objective:** Recognize human activities without wearable sensors, using WiFi CSI data.  
- **Dataset:** Custom multi-layout CSI dataset (5 and 15 activity classes).  
- **Input Representation:** CSI amplitude and phase preprocessed into RGB pseudocolor images.  
- **Models:**  
  - Baseline: 2D CNN  
  - Transformers: ViT-Tiny, ViT-Small, ViT-Base (from `timm` library, pretrained on ImageNet)  
- **Evaluation:** Accuracy, Precision, Recall, F1-Score, with augmentation experiments.

---

## Preprocessing Pipeline
1. Phase unwrapping to remove discontinuities.  
2. Gaussian smoothing to reduce noise.  
3. Outlier removal using ±3σ z-score and linear interpolation.  
4. Resampling to 256 temporal frames.  
5. Feature construction:  
   - Log-scaled amplitude  
   - Smoothed phase  
   - Temporal gradient of amplitude  
6. RGB stacking and normalization → `256 × 64 × 3` pseudocolor image.  
7. Resized to `224 × 224 × 3` for ViTs.

---

## Results Summary
- **CNN Baseline:** Provides solid performance on both public and custom datasets.  
- **Vision Transformers:** Outperformed CNNs under augmentation, with ViT-Base achieving the best overall accuracy.  
- **Cross-Layout Testing:** Accuracy dropped (~23% ViT, ~21–22% CNN), highlighting the challenge of layout-invariant recognition.

