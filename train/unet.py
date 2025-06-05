import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
import os
import random
from PIL import Image

# import sklearn # Not used in the provided snippet directly, but could be a dependency of bme1312
from bme1312 import (
    lab2 as lab,
)  # Assuming bme1312.lab2 and bme1312.evaluation are available
from bme1312.evaluation import get_DC, get_accuracy
from torchvision import transforms as T
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import (
    SummaryWriter,
)  # Not used in the final snippet, but present in original

plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

# Check available CUDA devices and use the first available one
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    # Use the last available GPU as per original code, or cuda:0 if only one.
    # If multiple GPUs, device_count-1 might not be optimal, often cuda:0 is default/first.
    # For consistency with original, using device_count-1.
    device_idx = device_count - 1 if device_count > 0 else 0
    device = torch.device(f"cuda:{device_idx}" if device_count > 0 else "cpu")
    print(f"CUDA devices available: {device_count}")
else:
    device = torch.device("cpu")
print("Using device: ", device)


def process_data():
    path = "datasets/cine_seg.npz"  # input the path of cine_seg.npz in your environment
    # Ensure the directory 'datasets' exists or the path is correct
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
    inputs = torch.Tensor(
        np.array(inputs, dtype=np.float32)
    )  # Ensure float32 for PyTorch
    labels = torch.Tensor(
        np.array(labels, dtype=np.float32)
    )  # Ensure float32 for PyTorch
    return inputs, labels


inputs, labels = process_data()
inputs = inputs.unsqueeze(1)  # Add channel dimension
labels = labels.unsqueeze(1)  # Add channel dimension
print("inputs shape: ", inputs.shape)
print("labels shape: ", labels.shape)


def convert_to_multi_labels(label_tensor):  # Renamed variable to avoid conflict
    device_label = label_tensor.device  # Use device of input tensor
    B, C, H, W = label_tensor.shape
    new_tensor = torch.zeros(
        (B, 3, H, W), device=device_label, dtype=torch.float32
    )  # Match dtype
    # Assuming label values are 0 (background), 85 (RV - Class 3), 170 (Myo - Class 2), 255 (LV - Class 1)
    mask1 = (label_tensor >= 250).squeeze(
        1
    )  # Class 1 (e.g., LV) - corresponds to channel 0
    mask2 = ((label_tensor >= 165) & (label_tensor < 250 - 1)).squeeze(
        1
    )  # Class 2 (e.g., Myo) - corresponds to channel 1
    mask3 = ((label_tensor >= 80) & (label_tensor < 165 - 1)).squeeze(
        1
    )  # Class 3 (e.g., RV) - corresponds to channel 2

    new_tensor[:, 0, :, :] = torch.where(  # Channel 0 is LV
        mask1,
        torch.ones_like(new_tensor[:, 0, :, :]),
        torch.zeros_like(new_tensor[:, 0, :, :]),
    )
    new_tensor[:, 1, :, :] = torch.where(  # Channel 1 is MYO
        mask2,
        torch.ones_like(new_tensor[:, 1, :, :]),
        torch.zeros_like(new_tensor[:, 1, :, :]),
    )
    new_tensor[:, 2, :, :] = torch.where(  # Channel 2 is RV
        mask3,
        torch.ones_like(new_tensor[:, 2, :, :]),
        torch.zeros_like(new_tensor[:, 2, :, :]),
    )
    return new_tensor


dataset = TensorDataset(inputs, labels)
batch_size = 32  # Reduced for potentially faster local testing if memory is an issue; user had 32
train_size = int(4 / 7 * len(dataset))
val_size = int(1 / 7 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Ensure reproducibility for splits if needed, by setting torch.manual_seed
torch.manual_seed(42)  # Example seed
train_set, val_set, test_set = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(
    val_set, batch_size=batch_size, shuffle=False
)  # Usually False for validation
dataloader_test = DataLoader(
    test_set, batch_size=batch_size, shuffle=False
)  # Usually False for testing


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


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            # For ConvTranspose2d, in_channels is the input to transpose conv.
            # The DoubleConv then takes in_channels (from skip connection + upsampled)
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.conv = DoubleConv(
            in_channels, out_channels
        )  # This in_channels is after concatenation

    def forward(self, x1, x2):  # x1 is from upsample, x2 is skip connection
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, C_base=64):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.C_base = C_base

        self.inc = DoubleConv(n_channels, C_base)
        self.down1 = Down(C_base, C_base * 2)
        self.down2 = Down(C_base * 2, C_base * 4)
        self.down3 = Down(C_base * 4, C_base * 8)
        self.down4 = Down(C_base * 8, C_base * 8)
        self.up1 = Up(C_base * 16, C_base * 4, bilinear)
        self.up2 = Up(C_base * 8, C_base * 2, bilinear)
        self.up3 = Up(C_base * 4, C_base, bilinear)
        self.up4 = Up(C_base * 2, C_base, bilinear)
        self.outc = nn.Conv2d(C_base, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class MyBinaryCrossEntropy(object):
    def __init__(self):
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss(reduction="mean")

    def __call__(self, pred_seg, seg_gt):
        pred_seg_probs = self.sigmoid(pred_seg)
        seg_gt_multilabel = convert_to_multi_labels(seg_gt)
        loss = self.bce(pred_seg_probs, seg_gt_multilabel)
        return loss


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

    class_names = ["LV", "MYO", "RV"]  # LV=index 0, MYO=index 1, RV=index 2

    for i, sample_idx in enumerate(random_indices):
        image, label = dataset_to_sample_from[sample_idx]
        image_batch = image.unsqueeze(0).to(device)
        label_batch = label.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_logits = model(image_batch)
            pred_masks_binary = torch.sigmoid(pred_logits) > 0.5

        gt_masks_multilabel = convert_to_multi_labels(label_batch)
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
            ax_img.imshow(image_np, cmap="gray")
            ax_img.set_title(f"Input Image\n(Context for {class_names[class_idx]})")
            ax_img.axis("off")

            ax_gt = axes[class_idx, 1]
            ax_gt.imshow(gt_masks_np[class_idx])
            ax_gt.set_title(f"Ground Truth - {class_names[class_idx]}")
            ax_gt.axis("off")

            ax_pred = axes[class_idx, 2]
            ax_pred.imshow(pred_masks_np[class_idx])
            ax_pred.set_title(f"Prediction - {class_names[class_idx]}")
            ax_pred.axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(model_output_path, f"sample_{i+1}_segmentation.png")
        plt.savefig(save_path)
        plt.close(fig)

    print(
        f"Saved {actual_num_samples} segmentation results for {model_name} in '{model_output_path}'"
    )


if not os.path.exists("result"):
    os.makedirs("result")

print("--- Training Baseline UNet ---")
net = UNet(n_channels=1, n_classes=3, C_base=32)
optimizer_baseline = torch.optim.Adam(net.parameters(), lr=0.01)
lr_scheduler_baseline = torch.optim.lr_scheduler.ExponentialLR(
    optimizer=optimizer_baseline, gamma=0.95
)
solver_baseline = lab.Solver(
    model=net,
    optimizer=optimizer_baseline,
    criterion=MyBinaryCrossEntropy(),
    lr_scheduler=lr_scheduler_baseline,
)
solver_baseline.train(
    epochs=50,
    data_loader=dataloader_train,
    val_loader=dataloader_val,
    img_name="baseline_unet",
)

print("\n--- Evaluating Baseline UNet ---")
dice_scores_baseline = []
net.to(device)
net.eval()
for images, labels_gt in dataloader_test:
    images = images.to(device)
    labels_gt = labels_gt.to(device)
    with torch.no_grad():
        preds_logits = net(images)
    preds_binary = torch.sigmoid(preds_logits) > 0.5
    labels_multilabel = convert_to_multi_labels(labels_gt)

    dice_lv_baseline_local = get_DC(
        preds_binary[:, 0, :, :], labels_multilabel[:, 0, :, :]
    )  # Index 0 is LV
    dice_myo_baseline_local = get_DC(
        preds_binary[:, 1, :, :], labels_multilabel[:, 1, :, :]
    )  # Index 1 is MYO
    dice_rv_baseline_local = get_DC(
        preds_binary[:, 2, :, :], labels_multilabel[:, 2, :, :]
    )  # Index 2 is RV
    dice_scores_baseline.append(
        (
            dice_lv_baseline_local.item(),
            dice_myo_baseline_local.item(),
            dice_rv_baseline_local.item(),
        )  # LV, MYO, RV
    )

mean_dice_lv_baseline = np.mean([score[0] for score in dice_scores_baseline])
std_dice_lv_baseline = np.std([score[0] for score in dice_scores_baseline])
mean_dice_myo_baseline = np.mean([score[1] for score in dice_scores_baseline])
std_dice_myo_baseline = np.std([score[1] for score in dice_scores_baseline])
mean_dice_rv_baseline = np.mean([score[2] for score in dice_scores_baseline])
std_dice_rv_baseline = np.std([score[2] for score in dice_scores_baseline])

print(
    f"LV Dice Coefficient: Mean={mean_dice_lv_baseline:.4f}, SD={std_dice_lv_baseline:.4f}"
)
print(
    f"MYO Dice Coefficient: Mean={mean_dice_myo_baseline:.4f}, SD={std_dice_myo_baseline:.4f}"
)
print(
    f"RV Dice Coefficient: Mean={mean_dice_rv_baseline:.4f}, SD={std_dice_rv_baseline:.4f}"
)

save_segmentation_results(
    net, test_set, device, "result", num_samples=3, model_name="baseline_unet"
)


class Up_NoShortcut(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up_NoShortcut, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)


class UNet_NoShortcut(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, C_base=64):
        super(UNet_NoShortcut, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.C_base = C_base
        self.inc = DoubleConv(n_channels, C_base)
        self.down1 = Down(C_base, C_base * 2)
        self.down2 = Down(C_base * 2, C_base * 4)
        self.down3 = Down(C_base * 4, C_base * 8)
        self.down4 = Down(C_base * 8, C_base * 8)
        self.up1 = Up_NoShortcut(C_base * 8, C_base * 4, bilinear)
        self.up2 = Up_NoShortcut(C_base * 4, C_base * 2, bilinear)
        self.up3 = Up_NoShortcut(C_base * 2, C_base, bilinear)
        self.up4 = Up_NoShortcut(C_base, C_base, bilinear)
        self.outc = nn.Conv2d(C_base, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.outc(x)
        return x


print("\n--- Training UNet No Shortcut ---")
net_no_shortcut = UNet_NoShortcut(n_channels=1, n_classes=3, C_base=32, bilinear=True)
optimizer_no_shortcut = torch.optim.Adam(net_no_shortcut.parameters(), lr=0.01)
lr_scheduler_no_shortcut = torch.optim.lr_scheduler.ExponentialLR(
    optimizer=optimizer_no_shortcut, gamma=0.95
)
solver_no_shortcut = lab.Solver(
    model=net_no_shortcut,
    optimizer=optimizer_no_shortcut,
    criterion=MyBinaryCrossEntropy(),
    lr_scheduler=lr_scheduler_no_shortcut,
)
solver_no_shortcut.train(
    epochs=50,
    data_loader=dataloader_train,
    val_loader=dataloader_val,
    img_name="no_shortcut_unet",
)

print("\n--- Evaluating UNet No Shortcut ---")
dice_scores_no_shortcut = []
net_no_shortcut.to(device)
net_no_shortcut.eval()
for images, labels_gt in dataloader_test:
    images = images.to(device)
    labels_gt = labels_gt.to(device)
    with torch.no_grad():
        preds_logits = net_no_shortcut(images)
    preds_binary = torch.sigmoid(preds_logits) > 0.5
    labels_multilabel = convert_to_multi_labels(labels_gt)

    dice_lv_no_shortcut = get_DC(
        preds_binary[:, 0, :, :], labels_multilabel[:, 0, :, :]
    )  # Index 0 is LV
    dice_myo_no_shortcut = get_DC(
        preds_binary[:, 1, :, :], labels_multilabel[:, 1, :, :]
    )  # Index 1 is MYO
    dice_rv_no_shortcut = get_DC(
        preds_binary[:, 2, :, :], labels_multilabel[:, 2, :, :]
    )  # Index 2 is RV
    dice_scores_no_shortcut.append(
        (
            dice_lv_no_shortcut.item(),
            dice_myo_no_shortcut.item(),
            dice_rv_no_shortcut.item(),
        )  # LV, MYO, RV
    )

mean_dice_lv_no_shortcut = np.mean([score[0] for score in dice_scores_no_shortcut])
std_dice_lv_no_shortcut = np.std([score[0] for score in dice_scores_no_shortcut])
mean_dice_myo_no_shortcut = np.mean([score[1] for score in dice_scores_no_shortcut])
std_dice_myo_no_shortcut = np.std([score[1] for score in dice_scores_no_shortcut])
mean_dice_rv_no_shortcut = np.mean([score[2] for score in dice_scores_no_shortcut])
std_dice_rv_no_shortcut = np.std([score[2] for score in dice_scores_no_shortcut])

print(
    f"LV Dice Coefficient Without Shortcut: Mean={mean_dice_lv_no_shortcut:.4f}, SD={std_dice_lv_no_shortcut:.4f}"
)
print(
    f"MYO Dice Coefficient Without Shortcut: Mean={mean_dice_myo_no_shortcut:.4f}, SD={std_dice_myo_no_shortcut:.4f}"
)
print(
    f"RV Dice Coefficient Without Shortcut: Mean={mean_dice_rv_no_shortcut:.4f}, SD={std_dice_rv_no_shortcut:.4f}"
)
save_segmentation_results(
    net_no_shortcut,
    test_set,
    device,
    "result",
    num_samples=3,
    model_name="no_shortcut_unet",
)


transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=50,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5,
        ),
    ]
)


class SegmentationDataset(data.Dataset):
    def __init__(self, inputs, labels, transform=None):
        self.inputs = inputs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        image = self.inputs[idx]
        label = self.labels[idx]
        if self.transform:
            all_stacked = torch.cat((image.unsqueeze(0), label.unsqueeze(0)), dim=0)
            combined = torch.cat([image, label], dim=0)  # Becomes (2, H, W)
            combined_transformed = self.transform(combined)
            image = combined_transformed[0, :, :].unsqueeze(0)  # Back to (1, H, W)
            label = combined_transformed[1, :, :].unsqueeze(0)  # Back to (1, H, W)

        return image, label


def extract_inputs_labels(dataset):
    inputs = []
    labels = []
    for data in dataset:
        input, label = data
        inputs.append(input)
        labels.append(label)
    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    return inputs, labels


inputs_train, labels_train = extract_inputs_labels(train_set)
inputs_val, labels_val = extract_inputs_labels(val_set)
inputs_test, labels_test = extract_inputs_labels(test_set)

train_dataset = SegmentationDataset(inputs_train, labels_train, transform=transform)
val_dataset = SegmentationDataset(
    inputs_val, labels_val, transform=None
)  # Val set usually not augmented

dataloader_train_aug = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
dataloader_val_aug = data.DataLoader(
    val_dataset, batch_size=32, shuffle=False
)  # Use non-augmented val_dataset

print("\n--- Training UNet with Data Augmentation ---")
net_data_aug = UNet(n_channels=1, n_classes=3, C_base=32)
optimizer_data_aug = torch.optim.Adam(net_data_aug.parameters(), lr=0.01)
lr_scheduler_data_aug = torch.optim.lr_scheduler.ExponentialLR(
    optimizer=optimizer_data_aug, gamma=0.95
)
solver_data_aug = lab.Solver(
    model=net_data_aug,
    optimizer=optimizer_data_aug,
    criterion=MyBinaryCrossEntropy(),
    lr_scheduler=lr_scheduler_data_aug,
)
solver_data_aug.train(
    epochs=50,
    data_loader=dataloader_train_aug,
    val_loader=dataloader_val_aug,  # Use non-augmented val loader
    img_name="baseline_unet_data_aug",
)

print("\n--- Evaluating UNet with Data Augmentation (Accuracy) ---")
accuracy_scores_data_aug = []
net_data_aug.to(device)
net_data_aug.eval()
for (
    images,
    labels_gt,
) in dataloader_test:  # Evaluate on original, non-augmented test set
    images = images.to(device)
    labels_gt = labels_gt.to(device)
    with torch.no_grad():
        preds_logits = net_data_aug(images)
    preds_binary = torch.sigmoid(preds_logits) > 0.5
    labels_multilabel = convert_to_multi_labels(labels_gt)

    accuracy_lv_data_aug_local = get_accuracy(
        preds_binary[:, 0, :, :], labels_multilabel[:, 0, :, :]
    )  # Index 0 is LV
    accuracy_myo_data_aug_local = get_accuracy(
        preds_binary[:, 1, :, :], labels_multilabel[:, 1, :, :]
    )  # Index 1 is MYO
    accuracy_rv_data_aug_local = get_accuracy(
        preds_binary[:, 2, :, :], labels_multilabel[:, 2, :, :]
    )  # Index 2 is RV
    accuracy_scores_data_aug.append(
        (
            accuracy_lv_data_aug_local,
            accuracy_myo_data_aug_local,
            accuracy_rv_data_aug_local,
        )
    )  # LV, MYO, RV

mean_accuracy_lv_aug = np.mean([score[0] for score in accuracy_scores_data_aug])
std_accuracy_lv_aug = np.std([score[0] for score in accuracy_scores_data_aug])
mean_accuracy_myo_aug = np.mean([score[1] for score in accuracy_scores_data_aug])
std_accuracy_myo_aug = np.std([score[1] for score in accuracy_scores_data_aug])
mean_accuracy_rv_aug = np.mean([score[2] for score in accuracy_scores_data_aug])
std_accuracy_rv_aug = np.std([score[2] for score in accuracy_scores_data_aug])

print(
    f"LV Accuracy (Data Aug): Mean={mean_accuracy_lv_aug:.4f}, SD={std_accuracy_lv_aug:.4f}"
)
print(
    f"MYO Accuracy (Data Aug): Mean={mean_accuracy_myo_aug:.4f}, SD={std_accuracy_myo_aug:.4f}"
)
print(
    f"RV Accuracy (Data Aug): Mean={mean_accuracy_rv_aug:.4f}, SD={std_accuracy_rv_aug:.4f}"
)

print("\n--- Evaluating UNet with Data Augmentation (Dice) ---")
dice_scores_data_aug = []
for images, labels_gt in dataloader_test:
    images = images.to(device)
    labels_gt = labels_gt.to(device)
    with torch.no_grad():
        preds_logits = net_data_aug(images)
    preds_binary = torch.sigmoid(preds_logits) > 0.5
    labels_multilabel = convert_to_multi_labels(labels_gt)

    dice_lv_data_aug = get_DC(
        preds_binary[:, 0, :, :], labels_multilabel[:, 0, :, :]
    )  # Index 0 is LV
    dice_myo_data_aug = get_DC(
        preds_binary[:, 1, :, :], labels_multilabel[:, 1, :, :]
    )  # Index 1 is MYO
    dice_rv_data_aug = get_DC(
        preds_binary[:, 2, :, :], labels_multilabel[:, 2, :, :]
    )  # Index 2 is RV
    dice_scores_data_aug.append(
        (
            dice_lv_data_aug.item(),
            dice_myo_data_aug.item(),
            dice_rv_data_aug.item(),
        )  # LV, MYO, RV
    )

mean_dice_lv_data_aug = np.mean([score[0] for score in dice_scores_data_aug])
std_dice_lv_data_aug = np.std([score[0] for score in dice_scores_data_aug])
mean_dice_myo_data_aug = np.mean([score[1] for score in dice_scores_data_aug])
std_dice_myo_data_aug = np.std([score[1] for score in dice_scores_data_aug])
mean_dice_rv_data_aug = np.mean([score[2] for score in dice_scores_data_aug])
std_dice_rv_data_aug = np.std([score[2] for score in dice_scores_data_aug])

print(
    f"LV Dice Coefficient With Data Augmentation: Mean={mean_dice_lv_data_aug:.4f}, SD={std_dice_lv_data_aug:.4f}"
)
print(
    f"MYO Dice Coefficient With Data Augmentation: Mean={mean_dice_myo_data_aug:.4f}, SD={std_dice_myo_data_aug:.4f}"
)
print(
    f"RV Dice Coefficient With Data Augmentation: Mean={mean_dice_rv_data_aug:.4f}, SD={std_dice_rv_data_aug:.4f}"
)
save_segmentation_results(
    net_data_aug,
    test_set,
    device,
    "result",
    num_samples=3,
    model_name="baseline_unet_data_aug",
)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred_logits, targets_gt):
        inputs_probs = self.sigmoid(pred_logits)
        targets_multilabel = convert_to_multi_labels(targets_gt)
        intersection = (inputs_probs * targets_multilabel).sum(dim=(2, 3))
        union_sum = inputs_probs.sum(dim=(2, 3)) + targets_multilabel.sum(dim=(2, 3))
        dice_coefficient_per_class = (2.0 * intersection + self.smooth) / (
            union_sum + self.smooth
        )
        dice_loss = 1 - dice_coefficient_per_class.mean()
        return dice_loss


print("\n--- Training UNet with Soft Dice Loss (No Data Augmentation) ---")
net_soft_dice = UNet(n_channels=1, n_classes=3, C_base=32)
optimizer_soft_dice = torch.optim.Adam(net_soft_dice.parameters(), lr=0.001)
lr_scheduler_soft_dice = torch.optim.lr_scheduler.ExponentialLR(
    optimizer=optimizer_soft_dice, gamma=0.95
)
dice_loss_fn = SoftDiceLoss()
solver_soft_dice = lab.Solver(
    model=net_soft_dice,
    optimizer=optimizer_soft_dice,
    criterion=dice_loss_fn,
    lr_scheduler=lr_scheduler_soft_dice,
)
solver_soft_dice.train(
    epochs=50,
    data_loader=dataloader_train,
    val_loader=dataloader_val,
    img_name="soft_dice_loss",
)

print("\n--- Evaluating UNet with Soft Dice Loss (Accuracy) ---")
accuracy_scores_soft_dice = []
net_soft_dice.to(device)
net_soft_dice.eval()
for images, labels_gt in dataloader_test:
    images = images.to(device)
    labels_gt = labels_gt.to(device)
    with torch.no_grad():
        preds_logits = net_soft_dice(images)
    preds_binary = torch.sigmoid(preds_logits) > 0.5
    labels_multilabel = convert_to_multi_labels(labels_gt)

    accuracy_lv_soft_dice = get_accuracy(
        preds_binary[:, 0, :, :], labels_multilabel[:, 0, :, :]
    )  # Index 0 is LV
    accuracy_myo_soft_dice = get_accuracy(
        preds_binary[:, 1, :, :], labels_multilabel[:, 1, :, :]
    )  # Index 1 is MYO
    accuracy_rv_soft_dice = get_accuracy(
        preds_binary[:, 2, :, :], labels_multilabel[:, 2, :, :]
    )  # Index 2 is RV
    accuracy_scores_soft_dice.append(
        (
            accuracy_lv_soft_dice,
            accuracy_myo_soft_dice,
            accuracy_rv_soft_dice,
        )  # LV, MYO, RV
    )

mean_accuracy_lv_soft_dice = np.mean([score[0] for score in accuracy_scores_soft_dice])
std_accuracy_lv_soft_dice = np.std([score[0] for score in accuracy_scores_soft_dice])
mean_accuracy_myo_soft_dice = np.mean([score[1] for score in accuracy_scores_soft_dice])
std_accuracy_myo_soft_dice = np.std([score[1] for score in accuracy_scores_soft_dice])
mean_accuracy_rv_soft_dice = np.mean([score[2] for score in accuracy_scores_soft_dice])
std_accuracy_rv_soft_dice = np.std([score[2] for score in accuracy_scores_soft_dice])

print(
    f"LV Accuracy With Soft Dice Loss: Mean={mean_accuracy_lv_soft_dice:.4f}, SD={std_accuracy_lv_soft_dice:.4f}"
)
print(
    f"MYO Accuracy With Soft Dice Loss: Mean={mean_accuracy_myo_soft_dice:.4f}, SD={std_accuracy_myo_soft_dice:.4f}"
)
print(
    f"RV Accuracy With Soft Dice Loss: Mean={mean_accuracy_rv_soft_dice:.4f}, SD={std_accuracy_rv_soft_dice:.4f}"
)

print(
    "\n--- Evaluating UNet with Soft Dice Loss (Dice) ---"
)  # Added this print statement for clarity
dice_scores_soft_dice = []
for images, labels_gt in dataloader_test:
    images = images.to(device)
    labels_gt = labels_gt.to(device)
    with torch.no_grad():
        preds_logits = net_soft_dice(images)
    preds_binary = torch.sigmoid(preds_logits) > 0.5
    labels_multilabel = convert_to_multi_labels(labels_gt)

    dice_lv_soft_dice = get_DC(
        preds_binary[:, 0, :, :], labels_multilabel[:, 0, :, :]
    )  # Index 0 is LV
    dice_myo_soft_dice = get_DC(
        preds_binary[:, 1, :, :], labels_multilabel[:, 1, :, :]
    )  # Index 1 is MYO
    dice_rv_soft_dice = get_DC(
        preds_binary[:, 2, :, :], labels_multilabel[:, 2, :, :]
    )  # Index 2 is RV
    dice_scores_soft_dice.append(
        (
            dice_lv_soft_dice.item(),
            dice_myo_soft_dice.item(),
            dice_rv_soft_dice.item(),
        )  # LV, MYO, RV
    )

mean_dice_lv_soft_dice = np.mean([score[0] for score in dice_scores_soft_dice])
std_dice_lv_soft_dice = np.std([score[0] for score in dice_scores_soft_dice])
mean_dice_myo_soft_dice = np.mean([score[1] for score in dice_scores_soft_dice])
std_dice_myo_soft_dice = np.std([score[1] for score in dice_scores_soft_dice])
mean_dice_rv_soft_dice = np.mean([score[2] for score in dice_scores_soft_dice])
std_dice_rv_soft_dice = np.std([score[2] for score in dice_scores_soft_dice])

print(
    f"LV Dice Coefficient With Soft Dice Loss: Mean={mean_dice_lv_soft_dice:.4f}, SD={std_dice_lv_soft_dice:.4f}"
)
print(
    f"MYO Dice Coefficient With Soft Dice Loss: Mean={mean_dice_myo_soft_dice:.4f}, SD={std_dice_myo_soft_dice:.4f}"
)
print(
    f"RV Dice Coefficient With Soft Dice Loss: Mean={mean_dice_rv_soft_dice:.4f}, SD={std_dice_rv_soft_dice:.4f}"
)
save_segmentation_results(
    net_soft_dice,
    test_set,
    device,
    "result",
    num_samples=3,
    model_name="soft_dice_loss_unet",
)

output_file_path = os.path.join("result", "output_results.txt")
print(f"\n--- Writing all results to {output_file_path} ---")
with open(output_file_path, "w") as file:
    file.write("--- Baseline UNet ---\n")
    file.write(
        f"LV Dice Coefficient: Mean={mean_dice_lv_baseline:.4f}, SD={std_dice_lv_baseline:.4f}\n"
    )
    file.write(
        f"MYO Dice Coefficient: Mean={mean_dice_myo_baseline:.4f}, SD={std_dice_myo_baseline:.4f}\n"
    )
    file.write(
        f"RV Dice Coefficient: Mean={mean_dice_rv_baseline:.4f}, SD={std_dice_rv_baseline:.4f}\n\n"
    )

    file.write("--- UNet No Shortcut ---\n")
    file.write(
        f"LV Dice Coefficient Without Shortcut: Mean={mean_dice_lv_no_shortcut:.4f}, SD={std_dice_lv_no_shortcut:.4f}\n"
    )
    file.write(
        f"MYO Dice Coefficient Without Shortcut: Mean={mean_dice_myo_no_shortcut:.4f}, SD={std_dice_myo_no_shortcut:.4f}\n"
    )
    file.write(
        f"RV Dice Coefficient Without Shortcut: Mean={mean_dice_rv_no_shortcut:.4f}, SD={std_dice_rv_no_shortcut:.4f}\n\n"
    )

    file.write("--- UNet with Data Augmentation ---\n")
    file.write(
        f"LV Accuracy (Data Aug): Mean={mean_accuracy_lv_aug:.4f}, SD={std_accuracy_lv_aug:.4f}\n"
    )
    file.write(
        f"MYO Accuracy (Data Aug): Mean={mean_accuracy_myo_aug:.4f}, SD={std_accuracy_myo_aug:.4f}\n"
    )
    file.write(
        f"RV Accuracy (Data Aug): Mean={mean_accuracy_rv_aug:.4f}, SD={std_accuracy_rv_aug:.4f}\n"
    )
    file.write(
        f"LV Dice Coefficient With Data Augmentation: Mean={mean_dice_lv_data_aug:.4f}, SD={std_dice_lv_data_aug:.4f}\n"
    )
    file.write(
        f"MYO Dice Coefficient With Data Augmentation: Mean={mean_dice_myo_data_aug:.4f}, SD={std_dice_myo_data_aug:.4f}\n"
    )
    file.write(
        f"RV Dice Coefficient With Data Augmentation: Mean={mean_dice_rv_data_aug:.4f}, SD={std_dice_rv_data_aug:.4f}\n\n"
    )

    file.write("--- UNet with Soft Dice Loss ---\n")
    file.write(
        f"LV Accuracy With Soft Dice Loss: Mean={mean_accuracy_lv_soft_dice:.4f}, SD={std_accuracy_lv_soft_dice:.4f}\n"
    )
    file.write(
        f"MYO Accuracy With Soft Dice Loss: Mean={mean_accuracy_myo_soft_dice:.4f}, SD={std_accuracy_myo_soft_dice:.4f}\n"
    )
    file.write(
        f"RV Accuracy With Soft Dice Loss: Mean={mean_accuracy_rv_soft_dice:.4f}, SD={std_accuracy_rv_soft_dice:.4f}\n"
    )
    file.write(
        f"LV Dice Coefficient With Soft Dice Loss: Mean={mean_dice_lv_soft_dice:.4f}, SD={std_dice_lv_soft_dice:.4f}\n"
    )
    file.write(
        f"MYO Dice Coefficient With Soft Dice Loss: Mean={mean_dice_myo_soft_dice:.4f}, SD={std_dice_myo_soft_dice:.4f}\n"
    )
    file.write(
        f"RV Dice Coefficient With Soft Dice Loss: Mean={mean_dice_rv_soft_dice:.4f}, SD={std_dice_rv_soft_dice:.4f}\n\n"
    )

print(f"All metrics saved to {output_file_path}")
print("Segmentation sample images saved in 'result/' subdirectories.")
