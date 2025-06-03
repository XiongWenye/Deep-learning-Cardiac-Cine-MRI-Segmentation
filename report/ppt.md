---
marp: true
theme: default
paginate: true
---

# Deep Learning for Cardiac Cine MRI Segmentation

**BME1312 Artificial Intelligence in Biomedical Imaging**
ShanghaiTech University
> Member: 熊闻野 夏博扬 杨人一 吴家兴 杨丰敏

---

## Overview

*   **Goal:** Segment key cardiac structures – LV, MYO, and RV
*   **Challenge:** Accurate and robust delineation of these structures, which can vary in shape and appearance.
*   **Approach:** U-Net based deep learning framework.
    1.  Baseline U-Net implementation.
    2.  Impact of removing U-Net skip connections.
    3.  Effect of data augmentation.
    4.  Comparison of Binary Cross-Entropy vs. Soft Dice Loss.
    5.  Improvements with Attention.
*   **Evaluation:** Dice Similarity Coefficient (DSC).

---

## Task (a): U-Net (Baseline)

<div align="center">
  <figure>
    <img src="..\result\network.png" alt="Network">
  </figure>
</div>

---

### Baseline Training Loss and validation Loss

<div align="center">
  <figure>
    <img src="..\result\baseline_unet.png" alt="Baseline Loss">
  </figure>
</div>

---

### Results: Dice Coefficients

| Structure | Mean Dice | Std. Dev. |
| :-------- | :-------- | :-------- |
| RV        | 0.9519    | 0.0086    |
| MYO       | 0.8734    | 0.0161    |
| LV        | 0.8920    | 0.0310    |

---

### Segmentation Examples (1/3)

<div align="center">
  <figure>
    <img src="..\result\for_ppt\baseline_LV.png" alt="Baseline Segmentation Example" width="1000">
    <figcaption><em> Baseline Segmentation Example LV</em></figcaption>
  </figure>
</div>

---

### Segmentation Examples (2/3)

<div align="center">
  <figure>
    <img src="..\result\for_ppt\baseline_MYO.png" alt="Baseline Segmentation Example" width="1000">
    <figcaption><em> Baseline Segmentation Example MYO</em></figcaption>
  </figure>
</div>

---

### Segmentation Examples (3/3)

<div align="center">
  <figure>
    <img src="..\result\for_ppt\baseline_RV.png" alt="Baseline Segmentation Example" width="1000">
    <figcaption><em> Baseline Segmentation Example RV</em></figcaption>
  </figure>
</div>

---

### Discussion - Baseline

*    **RV Segmentation**: Achieved the highest mean Dice score. This is often expected as the RV is typically a large, relatively well-defined structure with good contrast against surrounding tissues in many MRI sequences.
*    **LV Segmentation**: Also showed good performance. The LV cavity is usually clearly visible.
*    **MYO Segmentation**: Had the lowest mean Dice score. The myocardium is a thinner, more complex structure surrounding the LV, and its boundaries, especially with the LV cavity (endocardium) and epicardium, can be more challenging to delineate accurately, potentially leading to lower overlap scores.
*    The standard deviations are relatively small, indicating consistent performance across the test slices.

---


## Task (b): U-Net without Skip Connections

*   **Modification:** No skip connections in the U-Net architecture.
*   **Training:** Same as baseline (BCE Loss, lr=0.01, 50 epochs).
*   **Purpose:** Evaluate the importance of skip connections.

---

### Training Loss and validation Loss (No Short-cut)

  <div style="text-align: center; flex: 1;">
    <img src="../result/no_shortcut_unet.png" alt="Loss Curve for Baseline without shortcut" style="width:80%; max-height:400px; border: 1px solid #ccc;">
    <p style="font-size: 0.8em;"><em> Training and Validation Loss for Baseline U-Net without shortcut.</em></p>
  </div>

---

### Results: Dice Coefficients

| Structure | Baseline DSC | No Shortcut DSC |
| :-------- | :----------- | :-------------- |
| RV Mean   | **0.9519**   | 0.9260          |
| MYO Mean  | **0.8734**   | 0.8223          |
| LV Mean   | **0.8920**   | 0.8588          |
| RV std    | 0.0086       | 0.0111          |
| MYO std   | 0.0161       | 0.0168          |
| LV std    | 0.0310       | 0.0296          |

---

### Discussion - Impact of No Skip Connections

*   **Significant Drop in Performance:** All structures showed a noticeable decrease in DSC.
*   **Reason:** Skip connections provide high-resolution spatial information from the encoder to the decoder, crucial for accurate boundary localization. They also aid gradient flow.
*   **Conclusion:** Skip connections are vital for U-Net's segmentation accuracy in this task.

---

## Task (c): U-Net with Data Augmentation

*   **Network:** Baseline U-Net architecture.
*   **Augmentations (Training Set Only):**
    *   `RandomHorizontalFlip`, `RandomRotation(15°)`,
    *   `RandomAffine(degrees=50, translate=(0.1,0.1), scale=(0.9,1.1), shear=5)`.
*   **Implementation:** `SegmentationDataset` ensuring identical transforms for image and mask.
*   **Training:** BCE Loss, lr=0.01, 50 epochs.

---

### Training Loss and validation Loss (with Data Augmentation)

  <div style="text-align: center; flex: 1;">
    <img src="../result/baseline_unet_data_aug.png" alt="Loss Curve for Baseline with data aug" style="width:80%; max-height:400px; border: 1px solid #ccc;">
    <p style="font-size: 0.8em;"><em> Training and Validation Loss for Baseline with Data Augmentation.</em></p>
  </div>

---

### Results: Dice Coefficients
| Structure | Baseline DSC | Data Aug. DSC |
| :-------- | :----------- | :------------ |
| RV Mean   | **0.9519**   | 0.9276        |
| MYO Mean  | **0.8734**   | 0.8469        |
| LV Mean   | **0.8920**   | 0.8635        |
| RV std    | 0.0086       | 0.0107        |
| MYO std   | 0.0161       | 0.0149        |
| LV std    | 0.0310       | 0.0384        |


---

### Discussion - Impact of Data Augmentation

*   **DSC Decrease:** The specific augmentation strategy led to slightly lower Dice scores.
*   **Possible Reasons:**
    *   Some augmentations could have distorted anatomical structures, reducing the effectiveness of learning precise boundaries. Maybe the relative location of structures was altered too much.
*  **Conclusion:** The relative location of structures is crucial for segmentation tasks, and the specific augmentations used may not have been beneficial for this dataset. More careful selection or tuning of augmentations is needed.
---


## Task (d): U-Net with Soft Dice Loss

*   **Network:** Baseline U-Net architecture.
*   **Training Data:** Original Non-Augmented Training Set (The best).
*   **Loss Function:** `SoftDiceLoss`
*   **Optimizer:** Adam (lr=0.001), ExponentialLR scheduler.
*   **Training:** 50 epochs.

---

### Training Loss and validation Loss (With Soft Dice Loss)

  <div style="text-align: center; flex: 1;">
    <img src="../result/soft_dice_loss.png" alt="Loss Curve for Baseline with Soft Dice Loss" style="width:80%; max-height:400px; border: 1px solid #ccc;">
    <p style="font-size: 0.8em;"><em> Training and Validation Loss for Baseline with Soft Dice Loss.</em></p>
  </div>

---

### Results: Dice Coefficients

| Structure | Baseline with BCE Loss | Baseline with Soft Dice Loss |
| :-------- | :--------------------- | :--------------------------- |
| RV Mean   | 0.9519                 | **0.9566**                   |
| MYO Mean  | 0.8734                 | **0.8962**                   |
| LV Mean   | 0.8920                 | **0.8998**                   |
| RV std    | 0.0086                 | 0.0100                       |
| MYO std   | 0.0161                 | 0.0100                       |
| LV std    | 0.0310                 | 0.0371                       |

---

### Segmentation Examples (1/3)

<div align="center">
  <figure>
    <img src="..\result\for_ppt\soft_dice_loss_LV.png" alt="Baseline with Soft Dice Loss Segmentation Example" width="1000">
    <figcaption><em> Baseline with Soft Dice Loss Segmentation Example LV</em></figcaption>
  </figure>
</div>

---

### Segmentation Examples (2/3)

<div align="center">
  <figure>
    <img src="..\result\for_ppt\soft_dice_loss_MYO.png" alt="Baseline with Soft Dice Loss Segmentation Example" width="1000">
    <figcaption><em> Baseline with Soft Dice Loss Segmentation Example MYO</em></figcaption>
  </figure>
</div>

---

### Segmentation Examples (3/3)

<div align="center">
  <figure>
    <img src="..\result\for_ppt\soft_dice_loss_RV.png" alt="Baseline with Soft Dice Loss Segmentation Example" width="1000">
    <figcaption><em> Baseline with Soft Dice Loss Segmentation Example RV</em></figcaption>
  </figure>
</div>

---

### Discussion - Soft Dice Loss

*   When trained on the same non-augmented data, **Soft Dice Loss significantly outperformed BCE Loss** in terms of Dice Coefficient for all structures.
*   The improvement is most notable for MYO segmentation.
*   This suggests that directly optimizing a Dice-based metric is beneficial for this segmentation task.

---

## Task (e): Improvements
*   **Advanced UNet (Attention U-Net):**
  *   **Architecture:** Introduced `AttentionBlock` in the decoder's `Up` module.
    *   `AttentionBlock`: Computes attention coefficients by combining features from the decoder (gating signal) and encoder (skip connection), then applies these coefficients to the encoder features. This helps the model focus on relevant spatial regions during upsampling.
  *   **Loss Function:** `Soft Dice Loss`.
  *   **Optimizer:** Adam (lr=0.001), ExponentialLR scheduler.
  *   **Training:** 50 epochs.

---

### Results: Dice Coefficients
| Structure | Baseline with BCE Loss | Baseline with Soft Dice Loss | Attention U-Net |
| :-------- | :--------------------- | :--------------------------- | :-------------- |
| RV Mean   | 0.9519                 | 0.9566                       | **0.9588**      |
| MYO Mean  | 0.8734                 | 0.8962                       | **0.8967**      |
| LV Mean   | 0.8920                 | 0.8998                       | **0.9072**      |
| RV std    | 0.0086                 | 0.0100                       | 0.0086          |
| MYO std   | 0.0161                 | 0.0100                       | 0.0109          |
| LV std    | 0.0310                 | 0.0371                       | 0.0292          |

---

### Segmentation Examples (1/3)

<div align="center">
  <figure>
    <img src="..\result\for_ppt\attention_LV.png" alt="Attention U-Net Segmentation Example LV" width="1000">
    <figcaption><em> Attention U-Net Segmentation Example LV</em></figcaption>
  </figure>
</div>

---

### Segmentation Examples (2/3)

<div align="center">
  <figure>
    <img src="..\result\for_ppt\attention_MYO.png" alt="Attention U-Net Segmentation Example MYO" width="1000">
    <figcaption><em> Attention U-Net Segmentation Example MYO</em></figcaption>
  </figure>
</div>

---

### Segmentation Examples (3/3)

<div align="center">
  <figure>
    <img src="..\result\for_ppt\attention_RV.png" alt="Attention U-Net Segmentation Example RV" width="1000">
    <figcaption><em> Attention U-Net Segmentation Example RV</em></figcaption>
  </figure>
</div>

---

### Discussion - Attention U-Net

*   The Attention U-Net showed improved Dice scores compared to the baseline U-Net with BCE loss and the one with Soft Dice Loss.
*   This suggests that the attention mechanism effectively helps the model to focus on more complex structures or finer details, leading to better boundary delineation.
*   Accuracy scores are very high across all structures, which is common in segmentation tasks with large background areas. Dice coefficient remains a more informative metric for evaluating overlap.
---

## Overall Performance Summary (Dice Coefficients)
| Model                                    | RV Mean DSC | MYO Mean DSC | LV Mean DSC |
| :--------------------------------------- | :---------- | :----------- | :---------- |
| (a) Baseline U-Net (BCE)                 | 0.9519      | 0.8734       | 0.8920      |
| (b) U-Net No Shortcut (BCE)              | 0.9260      | 0.8223       | 0.8588      |
| (c) U-Net + Data Aug. (BCE)              | 0.9276      | 0.8469       | 0.8635      |
| (d) U-Net (Soft Dice Loss)               | 0.9566      | 0.8962       | 0.8998      |
| **(e) Attention U-Net (Soft Dice Loss)** | **0.9588**  | **0.8967**   | **0.9072**  |

---
## Conclusion & Future Work

*   **Key Findings:**
    *   U-Net with **Soft Dice Loss (trained on non-augmented data) yielded the best segmentation performance** (Dice scores).
    *   Skip connections are crucial.
    *   The specific data augmentation strategy tested did not improve Dice scores over the baseline non-augmented models.


---

## Thanks!
