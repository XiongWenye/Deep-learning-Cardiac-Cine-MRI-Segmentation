import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import (
    transforms,
)  # Kept for potential future use, but not active in this version
import os
import random
from PIL import Image

# import sklearn # Not used
from bme1312 import (
    lab2 as lab,
)  # Assuming bme1312.lab2 and bme1312.evaluation are available
from bme1312.evaluation import get_DC, get_accuracy

# from torchvision import transforms as T # Redundant with above
from torch.utils.data import TensorDataset, DataLoader

# from torch.utils.tensorboard import SummaryWriter # Not used

plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

# Check available CUDA devices and use the first available one
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    device_idx = device_count - 1 if device_count > 0 else 0
    device = torch.device(f"cuda:{device_idx}" if device_count > 0 else "cpu")
    print(f"CUDA devices available: {device_count}")
else:
    device = torch.device("cpu")
print("Using device: ", device)


def process_data():
    path = "datasets/cine_seg.npz"  # input the path of cine_seg.npz in your environment
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found at {path}. Please ensure the path is correct."
        )
    dataset = np.load(path, allow_pickle=True)
    files = dataset.files
    inputs = []
    labels = []
    for file in files:
        inputs.append(dataset[file][0])
        labels.append(dataset[file][1])
    inputs = torch.Tensor(np.array(inputs, dtype=np.float32))
    labels = torch.Tensor(np.array(labels, dtype=np.float32))
    return inputs, labels


inputs, labels = process_data()
inputs = inputs.unsqueeze(1)  # Add channel dimension
labels = labels.unsqueeze(1)  # Add channel dimension
print("inputs shape: ", inputs.shape)
print("labels shape: ", labels.shape)


def convert_to_multi_labels(label_tensor):
    device_label = label_tensor.device
    B, C, H, W = label_tensor.shape
    new_tensor = torch.zeros((B, 3, H, W), device=device_label, dtype=torch.float32)
    mask_lv = (label_tensor >= 250).squeeze(1)
    mask_myo = ((label_tensor >= 165) & (label_tensor < 250 - 1)).squeeze(1)
    mask_rv = ((label_tensor >= 80) & (label_tensor < 165 - 1)).squeeze(1)

    new_tensor[:, 0, :, :] = torch.where(
        mask_lv,
        torch.ones_like(new_tensor[:, 0, :, :]),
        torch.zeros_like(new_tensor[:, 0, :, :]),
    )
    new_tensor[:, 1, :, :] = torch.where(
        mask_myo,
        torch.ones_like(new_tensor[:, 1, :, :]),
        torch.zeros_like(new_tensor[:, 1, :, :]),
    )
    new_tensor[:, 2, :, :] = torch.where(
        mask_rv,
        torch.ones_like(new_tensor[:, 2, :, :]),
        torch.zeros_like(new_tensor[:, 2, :, :]),
    )
    return new_tensor


dataset = TensorDataset(inputs, labels)
batch_size = 32
train_size = int(4 / 7 * len(dataset))
val_size = int(1 / 7 * len(dataset))
test_size = len(dataset) - train_size - val_size

torch.manual_seed(42)
train_set, val_set, test_set = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(val_set, batch_size=batch_size, shuffle=False)
dataloader_test = DataLoader(test_set, batch_size=batch_size, shuffle=False)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.mpconv(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Attention Block
        Args:
            F_g: Number of channels in the gating signal (from decoder)
            F_l: Number of channels in the input signal (from encoder skip connection)
            F_int: Number of channels in the intermediate layer
        """
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: Gating signal from the decoder path (upsampled)
            x: Input signal from the encoder path (skip connection)
        Returns:
            Attention weighted input signal (x * attention_coefficients)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Up(nn.Module):  # Modified for Attention U-Net
    def __init__(
        self, in_channels_decoder, in_channels_encoder, out_channels, bilinear=True
    ):
        """
        Up-sampling block for Attention U-Net.
        Args:
            in_channels_decoder: Channels of the feature map from the lower decoder layer (to be upsampled).
            in_channels_encoder: Channels of the feature map from the corresponding encoder layer (skip connection).
            out_channels: Output channels of the DoubleConv layer.
            bilinear: Whether to use bilinear upsampling or transposed convolution.
        """
        super(Up, self).__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            upsampled_decoder_channels = in_channels_decoder
        else:
            # ConvTranspose2d output channels are typically half for U-Net like structures
            # to match encoder feature map sizes or a predefined scheme.
            # Here, we'll make it output in_channels_decoder // 2 channels.
            self.up = nn.ConvTranspose2d(
                in_channels_decoder, in_channels_decoder // 2, kernel_size=2, stride=2
            )
            upsampled_decoder_channels = in_channels_decoder // 2

        # Attention block
        # F_g: channels of the upsampled decoder feature map (g)
        # F_l: channels of the encoder feature map (x)
        # F_int: intermediate channels, often F_l // 2 or F_g // 2
        self.att = AttentionBlock(
            F_g=upsampled_decoder_channels,
            F_l=in_channels_encoder,
            F_int=in_channels_encoder // 2,
        )

        # DoubleConv after concatenation
        # Input channels to DoubleConv = channels from upsampled_decoder_map + channels from attention_weighted_encoder_map
        self.conv = DoubleConv(
            upsampled_decoder_channels + in_channels_encoder, out_channels
        )

    def forward(
        self, x_decoder, x_encoder
    ):  # x_decoder from lower layer, x_encoder is skip connection
        g = self.up(x_decoder)  # Upsampled decoder feature map

        # Pad 'g' if its spatial dimensions don't match 'x_encoder'
        # This is crucial for handling potential size mismatches from pooling/upsampling
        diffY = x_encoder.size()[2] - g.size()[2]
        diffX = x_encoder.size()[3] - g.size()[3]
        g = F.pad(g, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x_att = self.att(
            g, x_encoder
        )  # Apply attention to encoder features using g as context

        # Concatenate the upsampled decoder features (g) and attention-weighted encoder features (x_att)
        x_concat = torch.cat([g, x_att], dim=1)
        return self.conv(x_concat)


class AttentionUNet(nn.Module):
    def __init__(
        self, n_channels, n_classes, bilinear=True, C_base=32
    ):  # Using C_base=32 from your original baseline
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.C_base = C_base

        # Encoder
        self.inc = DoubleConv(n_channels, C_base)  # x1
        self.down1 = Down(C_base, C_base * 2)  # x2
        self.down2 = Down(C_base * 2, C_base * 4)  # x3
        self.down3 = Down(C_base * 4, C_base * 8)  # x4
        self.down4 = Down(C_base * 8, C_base * 8)  # x5 (bottleneck)

        # Decoder with Attention
        # Up(in_decoder_ch, in_encoder_ch, out_ch, bilinear)
        # Example for up1: x5 (C_base*8) is x_decoder, x4 (C_base*8) is x_encoder. Output has C_base*4.
        self.up1 = Up(C_base * 8, C_base * 8, C_base * 4, bilinear)
        self.up2 = Up(C_base * 4, C_base * 4, C_base * 2, bilinear)
        self.up3 = Up(C_base * 2, C_base * 2, C_base, bilinear)
        self.up4 = Up(C_base, C_base, C_base, bilinear)

        self.outc = nn.Conv2d(C_base, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # Bottleneck features

        # Decoder path using attention gates
        out = self.up1(x5, x4)  # Pass bottleneck (x5) and skip connection (x4)
        out = self.up2(out, x3)  # Pass output of up1 and skip connection (x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        logits = self.outc(out)
        return logits


class MyBinaryCrossEntropy(object):
    def __init__(self):
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss(reduction="mean")

    def __call__(self, pred_seg, seg_gt):
        pred_seg_probs = self.sigmoid(pred_seg)
        seg_gt_multilabel = convert_to_multi_labels(seg_gt)
        loss = self.bce(pred_seg_probs, seg_gt_multilabel)
        return loss


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred_seg_logits, seg_gt_raw):
        pred_seg_probs = self.sigmoid(pred_seg_logits)

        seg_gt_multilabel = convert_to_multi_labels(seg_gt_raw)

        if pred_seg_probs.shape != seg_gt_multilabel.shape:
            raise ValueError(
                f"Shape mismatch: pred_seg_probs {pred_seg_probs.shape} vs seg_gt_multilabel {seg_gt_multilabel.shape}"
            )

        intersection = (pred_seg_probs * seg_gt_multilabel).sum(dim=(2, 3))
        sum_probs = pred_seg_probs.sum(dim=(2, 3))
        sum_gt = seg_gt_multilabel.sum(dim=(2, 3))
        dice_coeff = (2.0 * intersection + self.smooth) / (
            sum_probs + sum_gt + self.smooth
        )
        dice_coeff_per_item = dice_coeff.mean(dim=1)

        dice_loss = 1 - dice_coeff_per_item.mean()

        return dice_loss


def save_segmentation_results(
    model, dataset_to_sample_from, device, output_base_dir, num_samples=3, model_name=""
):
    model_output_path = os.path.join(output_base_dir, model_name)
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path, exist_ok=True)

    model.eval()
    num_total_samples = len(dataset_to_sample_from)
    if num_total_samples == 0:
        print(f"Warning: Dataset for {model_name} is empty. Cannot save samples.")
        return
    actual_num_samples = min(num_samples, num_total_samples)
    random_indices = random.sample(range(num_total_samples), actual_num_samples)
    class_names = ["LV", "MYO", "RV"]  # As per cardiac segmentation, LV and RV swapped

    for i, sample_idx in enumerate(random_indices):
        image, label = dataset_to_sample_from[sample_idx]
        image_batch = image.unsqueeze(0).to(device)
        label_batch = label.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_logits = model(image_batch)
            pred_masks_binary = torch.sigmoid(pred_logits) > 0.5

        gt_masks_multilabel = convert_to_multi_labels(
            label_batch
        )  # Channel 0 is LV, Channel 2 is RV
        image_np = image_batch[0, 0].cpu().numpy()
        gt_masks_np = gt_masks_multilabel[0].cpu().numpy().astype(float)
        pred_masks_np = pred_masks_binary[0].cpu().numpy().astype(float)

        fig, axes = plt.subplots(
            len(class_names), 3, figsize=(12, 4 * len(class_names))
        )
        if len(class_names) == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle(f"{model_name} - Sample {i+1}", fontsize=16)

        for class_idx in range(len(class_names)):
            ax_img = axes[class_idx, 0]
            ax_img.imshow(image_np)
            ax_img.set_title(f"Input Image\n(Context for {class_names[class_idx]})")
            ax_img.axis("off")

            ax_gt = axes[class_idx, 1]
            ax_gt.imshow(
                gt_masks_np[class_idx]
            )  # gt_masks_np[0] is LV, gt_masks_np[2] is RV
            ax_gt.set_title(f"Ground Truth - {class_names[class_idx]}")
            ax_gt.axis("off")

            ax_pred = axes[class_idx, 2]
            ax_pred.imshow(
                pred_masks_np[class_idx]
            )  # pred_masks_np[0] is LV, pred_masks_np[2] is RV
            ax_pred.set_title(f"Prediction - {class_names[class_idx]}")
            ax_pred.axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(model_output_path, f"sample_{i+1}_segmentation.png")
        plt.savefig(save_path)
        plt.close(fig)

    print(
        f"Saved {actual_num_samples} segmentation results for {model_name} in '{model_output_path}'"
    )


# Ensure 'result' directory exists
if not os.path.exists("result"):
    os.makedirs("result")

# --- Training and Evaluating Advanced UNet (Attention U-Net) ---
print("--- Training Advanced UNet (Attention U-Net) ---")
# Initialize the AttentionUNet
# Using C_base=32 as in your original baseline UNet for fair comparison of architecture
attention_unet_model = AttentionUNet(
    n_channels=1, n_classes=3, bilinear=True, C_base=32
)
optimizer_att_unet = torch.optim.Adam(
    attention_unet_model.parameters(), lr=0.001
)  # You might want to tune LR
lr_scheduler_att_unet = torch.optim.lr_scheduler.ExponentialLR(
    optimizer=optimizer_att_unet, gamma=0.95
)

solver_att_unet = lab.Solver(
    model=attention_unet_model,
    optimizer=optimizer_att_unet,
    criterion=MyBinaryCrossEntropy(),
    lr_scheduler=lr_scheduler_att_unet,
)

solver_att_unet.train(
    epochs=50,  # Adjust epochs as needed
    data_loader=dataloader_train,
    val_loader=dataloader_val,
    img_name="attention_unet",  # Name for TensorBoard logs/saved models by Solver
)

print("\n--- Evaluating Advanced UNet (Attention U-Net) ---")
dice_scores_att_unet = []
accuracy_scores_att_unet = []  # Also calculating accuracy

attention_unet_model.to(device)
attention_unet_model.eval()

for images, labels_gt in dataloader_test:
    images = images.to(device)
    labels_gt = labels_gt.to(device)
    with torch.no_grad():
        preds_logits = attention_unet_model(images)

    preds_binary = torch.sigmoid(preds_logits) > 0.5  # (B, 3, H, W) boolean, then float
    labels_multilabel = convert_to_multi_labels(
        labels_gt
    )  # (B, 3, H, W) float. Channel 0 is LV, Channel 2 is RV.

    # Dice Scores
    # Channel 0 is LV, Channel 1 is MYO, Channel 2 is RV
    dice_lv = get_DC(preds_binary[:, 0, :, :], labels_multilabel[:, 0, :, :])
    dice_myo = get_DC(preds_binary[:, 1, :, :], labels_multilabel[:, 1, :, :])
    dice_rv = get_DC(preds_binary[:, 2, :, :], labels_multilabel[:, 2, :, :])
    dice_scores_att_unet.append(
        (dice_lv.item(), dice_myo.item(), dice_rv.item())
    )  # Order: LV, MYO, RV

    # Accuracy Scores
    acc_lv = get_accuracy(preds_binary[:, 0, :, :], labels_multilabel[:, 0, :, :])
    acc_myo = get_accuracy(preds_binary[:, 1, :, :], labels_multilabel[:, 1, :, :])
    acc_rv = get_accuracy(preds_binary[:, 2, :, :], labels_multilabel[:, 2, :, :])
    accuracy_scores_att_unet.append((acc_lv, acc_myo, acc_rv))  # Order: LV, MYO, RV

# score[0] is LV, score[1] is MYO, score[2] is RV
mean_dice_lv_att = np.mean([score[0] for score in dice_scores_att_unet])
std_dice_lv_att = np.std([score[0] for score in dice_scores_att_unet])
mean_dice_myo_att = np.mean([score[1] for score in dice_scores_att_unet])
std_dice_myo_att = np.std([score[1] for score in dice_scores_att_unet])
mean_dice_rv_att = np.mean([score[2] for score in dice_scores_att_unet])
std_dice_rv_att = np.std([score[2] for score in dice_scores_att_unet])

print(
    f"Attention U-Net - LV Dice Coefficient: Mean={mean_dice_lv_att:.4f}, SD={std_dice_lv_att:.4f}"
)
print(
    f"Attention U-Net - MYO Dice Coefficient: Mean={mean_dice_myo_att:.4f}, SD={std_dice_myo_att:.4f}"
)
print(
    f"Attention U-Net - RV Dice Coefficient: Mean={mean_dice_rv_att:.4f}, SD={std_dice_rv_att:.4f}"
)

mean_acc_lv_att = np.mean([score[0] for score in accuracy_scores_att_unet])
std_acc_lv_att = np.std([score[0] for score in accuracy_scores_att_unet])
mean_acc_myo_att = np.mean([score[1] for score in accuracy_scores_att_unet])
std_acc_myo_att = np.std([score[1] for score in accuracy_scores_att_unet])
mean_acc_rv_att = np.mean([score[2] for score in accuracy_scores_att_unet])
std_acc_rv_att = np.std([score[2] for score in accuracy_scores_att_unet])

print(
    f"Attention U-Net - LV Accuracy: Mean={mean_acc_lv_att:.4f}, SD={std_acc_lv_att:.4f}"
)
print(
    f"Attention U-Net - MYO Accuracy: Mean={mean_acc_myo_att:.4f}, SD={std_acc_myo_att:.4f}"
)
print(
    f"Attention U-Net - RV Accuracy: Mean={mean_acc_rv_att:.4f}, SD={std_acc_rv_att:.4f}"
)

# Save segmentation results for Attention U-Net
save_segmentation_results(
    attention_unet_model,
    test_set,
    device,
    "result",
    num_samples=3,
    model_name="attention_unet",
)

# --- Writing Results to File ---
output_file_path = os.path.join("result", "output_results_attention_unet.txt")
print(f"\n--- Writing Attention U-Net results to {output_file_path} ---")

with open(output_file_path, "w") as file:
    file.write("--- Advanced UNet (Attention U-Net) ---\n")
    file.write(
        f"LV Dice Coefficient: Mean={mean_dice_lv_att:.4f}, SD={std_dice_lv_att:.4f}\n"
    )
    file.write(
        f"MYO Dice Coefficient: Mean={mean_dice_myo_att:.4f}, SD={std_dice_myo_att:.4f}\n"
    )
    file.write(
        f"RV Dice Coefficient: Mean={mean_dice_rv_att:.4f}, SD={std_dice_rv_att:.4f}\n\n"
    )

    file.write(f"LV Accuracy: Mean={mean_acc_lv_att:.4f}, SD={std_acc_lv_att:.4f}\n")
    file.write(f"MYO Accuracy: Mean={mean_acc_myo_att:.4f}, SD={std_acc_myo_att:.4f}\n")
    file.write(f"RV Accuracy: Mean={mean_acc_rv_att:.4f}, SD={std_acc_rv_att:.4f}\n")

print(f"Attention U-Net metrics saved to {output_file_path}")
print(
    "Segmentation sample images for Attention U-Net saved in 'result/attention_unet' subdirectory."
)
