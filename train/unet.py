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
from bme1312 import lab2 as lab # Assuming bme1312.lab2 and bme1312.evaluation are available
from bme1312.evaluation import get_DC,get_accuracy
from torchvision import transforms as T
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter # Not used in the final snippet, but present in original

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Check available CUDA devices and use the first available one
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    # Use the last available GPU as per original code, or cuda:0 if only one.
    # If multiple GPUs, device_count-1 might not be optimal, often cuda:0 is default/first.
    # For consistency with original, using device_count-1.
    device_idx = device_count - 1 if device_count > 0 else 0
    device = torch.device(f'cuda:{device_idx}' if device_count > 0 else 'cpu')
    print(f"CUDA devices available: {device_count}")
else:
    device = torch.device('cpu')
print("Using device: ", device)

def process_data():
    path="datasets/cine_seg.npz" # input the path of cine_seg.npz in your environment
    # Ensure the directory 'datasets' exists or the path is correct
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at {path}. Please ensure the path is correct.")
    dataset=np.load(path,allow_pickle=True)
    files=dataset.files
    inputs=[]
    labels=[]
    for file in files:
        inputs.append(dataset[file][0])
        labels.append(dataset[file][1])
    inputs = torch.Tensor(np.array(inputs, dtype=np.float32)) # Ensure float32 for PyTorch
    labels = torch.Tensor(np.array(labels, dtype=np.float32)) # Ensure float32 for PyTorch
    return inputs, labels

inputs, labels = process_data()
inputs = inputs.unsqueeze(1)  # Add channel dimension
labels = labels.unsqueeze(1)  # Add channel dimension
print("inputs shape: ", inputs.shape)
print("labels shape: ", labels.shape)


def convert_to_multi_labels(label_tensor): # Renamed variable to avoid conflict
    device_label = label_tensor.device # Use device of input tensor
    B, C, H, W = label_tensor.shape
    new_tensor = torch.zeros((B, 3, H, W), device=device_label, dtype=torch.float32) # Match dtype
    # Assuming label values are 0 (background), 85 (class 3), 170 (class 2), 255 (class 1)
    mask1 = (label_tensor >= 250).squeeze(1)  # Class 1 (e.g., RV)
    mask2 = ((label_tensor >= 165) & (label_tensor < 250-1)).squeeze(1) # Class 2 (e.g., Myo)
    mask3 = ((label_tensor >= 80) & (label_tensor < 165-1)).squeeze(1)   # Class 3 (e.g., LV)
    
    # Using torch.where requires boolean masks. Squeeze can remove C if C=1.
    # Ensure masks have the correct shape for broadcasting if necessary, though here it should be fine.

    new_tensor[:, 0, :, :] = torch.where(mask1, torch.ones_like(new_tensor[:, 0, :, :]), torch.zeros_like(new_tensor[:, 0, :, :]))
    new_tensor[:, 1, :, :] = torch.where(mask2, torch.ones_like(new_tensor[:, 1, :, :]), torch.zeros_like(new_tensor[:, 1, :, :]))
    new_tensor[:, 2, :, :] = torch.where(mask3, torch.ones_like(new_tensor[:, 2, :, :]), torch.zeros_like(new_tensor[:, 2, :, :]))
    return new_tensor

dataset = TensorDataset(inputs, labels)
batch_size = 32 # Reduced for potentially faster local testing if memory is an issue; user had 32
train_size = int(4/7 * len(dataset))
val_size = int(1/7 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Ensure reproducibility for splits if needed, by setting torch.manual_seed
torch.manual_seed(42) # Example seed
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(val_set, batch_size=batch_size, shuffle=False) # Usually False for validation
dataloader_test =  DataLoader(test_set, batch_size=batch_size, shuffle=False) # Usually False for testing


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # For ConvTranspose2d, in_channels is the input to transpose conv.
            # The DoubleConv then takes in_channels (from skip connection + upsampled)
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels) # This in_channels is after concatenation

    def forward(self, x1, x2): # x1 is from upsample, x2 is skip connection
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
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
        self.down1 = Down(C_base, C_base*2)
        self.down2 = Down(C_base*2, C_base*4)
        self.down3 = Down(C_base*4, C_base*8)
        self.down4 = Down(C_base*8, C_base*8) # Original was C_base*8, C_base*8, for Up1 C_base*16 suggests this should be C_base*16 or Up takes C_base*8 (from down4) + C_base*8 (from x4)
                                            # Assuming down4 output is C_base*8.
        
        # If down4 outputs C_base*8, then Up1 input from x5 is C_base*8.
        # Skip connection x4 is C_base*8. Concatenated they are C_base*16.
        self.up1 = Up(C_base*16, C_base*4, bilinear) # Correct: 1024 -> 512 (if C_base=64). C_base*8 + C_base*8 = C_base*16
        self.up2 = Up(C_base*8, C_base*2, bilinear)  # C_base*4 + C_base*4 = C_base*8
        self.up3 = Up(C_base*4, C_base, bilinear)    # C_base*2 + C_base*2 = C_base*4
        self.up4 = Up(C_base*2, C_base, bilinear)    # C_base   + C_base   = C_base*2
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
        # BCELoss expects target to be in [0,1] and same shape as input.
        # convert_to_multi_labels already makes target suitable.
        self.bce = nn.BCELoss(reduction='mean') 

    def __call__(self, pred_seg, seg_gt): # pred_seg are logits
        pred_seg_probs = self.sigmoid(pred_seg) # Convert logits to probabilities
        # seg_gt comes from dataloader, shape (B, 1, H, W) with values like 0, 85, 170, 255
        seg_gt_multilabel = convert_to_multi_labels(seg_gt) # Convert to (B, 3, H, W) binary masks
        loss = self.bce(pred_seg_probs, seg_gt_multilabel)
        return loss

# Define the new function to save segmentation results
def save_segmentation_results(model, dataset_to_sample_from, device, output_base_dir, num_samples=3, model_name=""):
    """
    Saves num_samples random segmentation results from the model.

    Args:
        model: The trained PyTorch model.
        dataset_to_sample_from: The torch.utils.data.Dataset to sample from (e.g., test_set).
        device: The device to run inference on.
        output_base_dir: The base directory to save results (e.g., "result").
        num_samples: Number of random samples to visualize and save.
        model_name: Name of the model, used to create a subdirectory under output_base_dir.
    """
    model_output_path = os.path.join(output_base_dir, model_name)
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path, exist_ok=True)

    model.eval() # Ensure model is in evaluation mode
    
    num_total_samples = len(dataset_to_sample_from)
    if num_total_samples == 0:
        print(f"Warning: Dataset for {model_name} is empty. Cannot save samples.")
        return
        
    actual_num_samples = min(num_samples, num_total_samples)
    
    random_indices = random.sample(range(num_total_samples), actual_num_samples)

    class_names = ['RV', 'MYO', 'LV'] # As per typical cardiac segmentation

    for i, sample_idx in enumerate(random_indices):
        image, label = dataset_to_sample_from[sample_idx] # image: (1,H,W), label: (1,H,W)

        # Add batch dimension and move to device
        image_batch = image.unsqueeze(0).to(device) # (1, 1, H, W)
        label_batch = label.unsqueeze(0).to(device) # (1, 1, H, W)

        with torch.no_grad():
            pred_logits = model(image_batch) # (1, 3, H, W)
            pred_masks_binary = torch.sigmoid(pred_logits) > 0.5 # (1, 3, H, W), bool then cast to float by ops
        
        gt_masks_multilabel = convert_to_multi_labels(label_batch) # (1, 3, H, W)

        # Prepare for plotting: move to CPU, convert to NumPy
        # Input image: (1, 1, H, W) -> (H, W)
        image_np = image_batch[0, 0].cpu().numpy()
        
        # GT masks: (1, 3, H, W) -> (3, H, W)
        gt_masks_np = gt_masks_multilabel[0].cpu().numpy().astype(float)
        
        # Predicted masks: (1, 3, H, W) -> (3, H, W)
        pred_masks_np = pred_masks_binary[0].cpu().numpy().astype(float)

        fig, axes = plt.subplots(len(class_names), 3, figsize=(12, 4 * len(class_names))) # (width, height)
        if len(class_names) == 1: # Adjust for single class case if axes is not a 2D array
             axes = axes.reshape(1, -1)

        fig.suptitle(f'{model_name} - Sample {i+1}', fontsize=16)

        for class_idx in range(len(class_names)):
            # Input Image (repeated for each class row for context)
            ax_img = axes[class_idx, 0]
            ax_img.imshow(image_np, cmap='gray')
            ax_img.set_title(f"Input Image\n(Context for {class_names[class_idx]})")
            ax_img.axis('off')

            # Ground Truth Mask
            ax_gt = axes[class_idx, 1]
            ax_gt.imshow(gt_masks_np[class_idx], cmap='viridis', vmin=0, vmax=1)
            ax_gt.set_title(f"Ground Truth - {class_names[class_idx]}")
            ax_gt.axis('off')

            # Predicted Mask
            ax_pred = axes[class_idx, 2]
            ax_pred.imshow(pred_masks_np[class_idx], cmap='viridis', vmin=0, vmax=1)
            ax_pred.set_title(f"Prediction - {class_names[class_idx]}")
            ax_pred.axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
        save_path = os.path.join(model_output_path, f'sample_{i+1}_segmentation.png')
        plt.savefig(save_path)
        plt.close(fig)
        
    print(f"Saved {actual_num_samples} segmentation results for {model_name} in '{model_output_path}'")

# Ensure 'result' directory exists
if not os.path.exists('result'):
    os.makedirs('result')
# --- Baseline UNet ---
print("--- Training Baseline UNet ---")
net = UNet(n_channels=1, n_classes=3, C_base=32) # C_base=32 as per user's code
optimizer_baseline = torch.optim.Adam(net.parameters(), lr=0.01)
lr_scheduler_baseline = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_baseline, gamma=0.95)

solver_baseline = lab.Solver(
    model=net,
    optimizer=optimizer_baseline,
    criterion=MyBinaryCrossEntropy(),
    lr_scheduler=lr_scheduler_baseline,
)

solver_baseline.train(
    epochs=50, # Consider fewer epochs for quick testing, e.g., 2-5
    data_loader=dataloader_train,
    val_loader=dataloader_val,
    img_name='baseline_unet' # This name is used by lab.Solver, e.g. for TensorBoard
)

print("\n--- Evaluating Baseline UNet ---")
dice_scores_baseline = []
net.to(device) # Ensure model is on the correct device for evaluation
net.eval() # Set model to evaluation mode

for images, labels_gt in dataloader_test:
    images = images.to(device)  
    labels_gt = labels_gt.to(device) 
    with torch.no_grad():
        preds_logits = net(images)
    preds_binary = torch.sigmoid(preds_logits) > 0.5
    labels_multilabel = convert_to_multi_labels(labels_gt)

    # Ensure preds_binary and labels_multilabel are (B, H, W) for get_DC
    dice_rv = get_DC(preds_binary[:, 0, :, :], labels_multilabel[:, 0, :, :])
    dice_myo = get_DC(preds_binary[:, 1, :, :], labels_multilabel[:, 1, :, :])
    dice_lv = get_DC(preds_binary[:, 2, :, :], labels_multilabel[:, 2, :, :])
    dice_scores_baseline.append((dice_rv.item(), dice_myo.item(), dice_lv.item())) # .item() to get float

mean_dice_rv = np.mean([score[0] for score in dice_scores_baseline])
std_dice_rv = np.std([score[0] for score in dice_scores_baseline])
mean_dice_myo = np.mean([score[1] for score in dice_scores_baseline])
std_dice_myo = np.std([score[1] for score in dice_scores_baseline])
mean_dice_lv = np.mean([score[2] for score in dice_scores_baseline])
std_dice_lv = np.std([score[2] for score in dice_scores_baseline])

print(f'RV Dice Coefficient: Mean={mean_dice_rv:.4f}, SD={std_dice_rv:.4f}')
print(f'MYO Dice Coefficient: Mean={mean_dice_myo:.4f}, SD={std_dice_myo:.4f}')
print(f'LV Dice Coefficient: Mean={mean_dice_lv:.4f}, SD={std_dice_lv:.4f}')

# Save segmentation results for baseline UNet
save_segmentation_results(net, test_set, device, 'result', num_samples=3, model_name='baseline_unet')


# --- UNet No Shortcut ---
class Up_NoShortcut(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up_NoShortcut, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After upsampling, input to DoubleConv is in_channels (from previous layer)
            self.conv = DoubleConv(in_channels, out_channels) 
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x1): # x1 is the input from the previous layer in the encoder
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
        self.down1 = Down(C_base, C_base*2)
        self.down2 = Down(C_base*2, C_base*4)
        self.down3 = Down(C_base*4, C_base*8)
        self.down4 = Down(C_base*8, C_base*8) # Output C_base*8

        self.up1 = Up_NoShortcut(C_base*8, C_base*4, bilinear) # Input from down4 (C_base*8)
        self.up2 = Up_NoShortcut(C_base*4, C_base*2, bilinear) # Input from up1 (C_base*4)
        self.up3 = Up_NoShortcut(C_base*2, C_base, bilinear)   # Input from up2 (C_base*2)
        self.up4 = Up_NoShortcut(C_base, C_base, bilinear)     # Input from up3 (C_base)
        self.outc = nn.Conv2d(C_base, n_classes, kernel_size=1)


    def forward(self, x):
        x1 = self.inc(x)    # C_base
        x2 = self.down1(x1) # C_base*2
        x3 = self.down2(x2) # C_base*4
        x4 = self.down3(x3) # C_base*8
        x5 = self.down4(x4) # C_base*8
        
        x = self.up1(x5)    # C_base*4
        x = self.up2(x)     # C_base*2
        x = self.up3(x)     # C_base
        x = self.up4(x)     # C_base
        x = self.outc(x)
        return x

print("\n--- Training UNet No Shortcut ---")
net_no_shortcut = UNet_NoShortcut(n_channels=1, n_classes=3, C_base=32, bilinear=True) # Explicitly setting bilinear
optimizer_no_shortcut = torch.optim.Adam(net_no_shortcut.parameters(), lr=0.01)
lr_scheduler_no_shortcut = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_no_shortcut, gamma=0.95) # Use new optimizer variable

solver_no_shortcut = lab.Solver(
    model=net_no_shortcut,
    optimizer=optimizer_no_shortcut, # Pass the correct optimizer
    criterion=MyBinaryCrossEntropy(),
    lr_scheduler=lr_scheduler_no_shortcut 
)

solver_no_shortcut.train(
    epochs=50, # Fewer epochs for testing
    data_loader=dataloader_train,
    val_loader=dataloader_val,
    img_name='no_shortcut_unet'
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

    dice_rv_no_shortcut = get_DC(preds_binary[:, 0, :, :], labels_multilabel[:, 0, :, :])
    dice_myo_no_shortcut = get_DC(preds_binary[:, 1, :, :], labels_multilabel[:, 1, :, :])
    dice_lv_no_shortcut = get_DC(preds_binary[:, 2, :, :], labels_multilabel[:, 2, :, :])
    dice_scores_no_shortcut.append((dice_rv_no_shortcut.item(), dice_myo_no_shortcut.item(), dice_lv_no_shortcut.item()))

mean_dice_rv_no_shortcut = np.mean([score[0] for score in dice_scores_no_shortcut])
std_dice_rv_no_shortcut = np.std([score[0] for score in dice_scores_no_shortcut])
mean_dice_myo_no_shortcut = np.mean([score[1] for score in dice_scores_no_shortcut])
std_dice_myo_no_shortcut = np.std([score[1] for score in dice_scores_no_shortcut])
mean_dice_lv_no_shortcut = np.mean([score[2] for score in dice_scores_no_shortcut])
std_dice_lv_no_shortcut = np.std([score[2] for score in dice_scores_no_shortcut])

print(f'RV Dice Coefficient Without Shortcut: Mean={mean_dice_rv_no_shortcut:.4f}, SD={std_dice_rv_no_shortcut:.4f}')
print(f'MYO Dice Coefficient Without Shortcut: Mean={mean_dice_myo_no_shortcut:.4f}, SD={std_dice_myo_no_shortcut:.4f}')
print(f'LV Dice Coefficient Without Shortcut: Mean={mean_dice_lv_no_shortcut:.4f}, SD={std_dice_lv_no_shortcut:.4f}')

# Save segmentation results for UNet No Shortcut
save_segmentation_results(net_no_shortcut, test_set, device, 'result', num_samples=3, model_name='no_shortcut_unet')


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(
        degrees=50,
        translate=(0.1, 0.1),  # Small translations
        scale=(0.9, 1.1),      # Slight scaling
        shear=5                # Small shear transformations
    )
])

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
            all = torch.stack((image, label), dim = 0)
            all = self.transform(all)
            image = all[0]
            label = all[1]
            
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
val_dataset = SegmentationDataset(inputs_val, labels_val, transform=None)

dataloader_train_aug = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
dataloader_val_aug = data.DataLoader(val_dataset, batch_size=32, shuffle=False)

print("\n--- Training UNet with Data Augmentation ---")
net_data_aug = UNet(n_channels=1, n_classes=3, C_base=32)
optimizer_data_aug = torch.optim.Adam(net_data_aug.parameters(), lr=0.01)
lr_scheduler_data_aug = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_data_aug, gamma=0.95)

solver_data_aug = lab.Solver( # Renamed solver variable
    model=net_data_aug,
    optimizer=optimizer_data_aug,
    criterion=MyBinaryCrossEntropy(),
    lr_scheduler=lr_scheduler_data_aug,
)

solver_data_aug.train(
    epochs=50, # Fewer epochs for testing
    data_loader=dataloader_train_aug,
    val_loader=dataloader_val_aug, # Use non-augmented val loader
    img_name='baseline_unet_data_aug'
)

print("\n--- Evaluating UNet with Data Augmentation (Accuracy) ---")
accuracy_scores_data_aug = [] # Renamed
net_data_aug.to(device)
net_data_aug.eval()

for images, labels_gt in dataloader_test: # Evaluate on original, non-augmented test set
    images = images.to(device)  
    labels_gt = labels_gt.to(device) 
    with torch.no_grad():
        preds_logits = net_data_aug(images)
    preds_binary = torch.sigmoid(preds_logits) > 0.5
    labels_multilabel = convert_to_multi_labels(labels_gt)

    # get_accuracy expects (B, H, W) or (H, W) inputs.
    # Ensure preds_binary and labels_multilabel are suitable.
    accuracy_rv = get_accuracy(preds_binary[:, 0, :, :], labels_multilabel[:, 0, :, :])
    accuracy_myo = get_accuracy(preds_binary[:, 1, :, :], labels_multilabel[:, 1, :, :])
    accuracy_lv = get_accuracy(preds_binary[:, 2, :, :], labels_multilabel[:, 2, :, :])
    accuracy_scores_data_aug.append((accuracy_rv, accuracy_myo, accuracy_lv))

mean_accuracy_rv_aug = np.mean([score[0] for score in accuracy_scores_data_aug]) # Renamed
std_accuracy_rv_aug = np.std([score[0] for score in accuracy_scores_data_aug])
mean_accuracy_myo_aug = np.mean([score[1] for score in accuracy_scores_data_aug])
std_accuracy_myo_aug = np.std([score[1] for score in accuracy_scores_data_aug])
mean_accuracy_lv_aug = np.mean([score[2] for score in accuracy_scores_data_aug])
std_accuracy_lv_aug = np.std([score[2] for score in accuracy_scores_data_aug])

print(f'RV Accuracy (Data Aug): Mean={mean_accuracy_rv_aug:.4f}, SD={std_accuracy_rv_aug:.4f}')
print(f'MYO Accuracy (Data Aug): Mean={mean_accuracy_myo_aug:.4f}, SD={std_accuracy_myo_aug:.4f}')
print(f'LV Accuracy (Data Aug): Mean={mean_accuracy_lv_aug:.4f}, SD={std_accuracy_lv_aug:.4f}')

print("\n--- Evaluating UNet with Data Augmentation (Dice) ---")
dice_scores_data_aug = []
# net_data_aug is already on device and in eval mode

for images, labels_gt in dataloader_test:
    images = images.to(device)  
    labels_gt = labels_gt.to(device) 
    with torch.no_grad():
        preds_logits = net_data_aug(images)
    preds_binary = torch.sigmoid(preds_logits) > 0.5
    labels_multilabel = convert_to_multi_labels(labels_gt)

    dice_rv_data_aug = get_DC(preds_binary[:, 0, :, :], labels_multilabel[:, 0, :, :])
    dice_myo_data_aug = get_DC(preds_binary[:, 1, :, :], labels_multilabel[:, 1, :, :])
    dice_lv_data_aug = get_DC(preds_binary[:, 2, :, :], labels_multilabel[:, 2, :, :])
    dice_scores_data_aug.append((dice_rv_data_aug.item(), dice_myo_data_aug.item(), dice_lv_data_aug.item()))

mean_dice_rv_data_aug = np.mean([score[0] for score in dice_scores_data_aug])
std_dice_rv_data_aug= np.std([score[0] for score in dice_scores_data_aug])
mean_dice_myo_data_aug = np.mean([score[1] for score in dice_scores_data_aug])
std_dice_myo_data_aug = np.std([score[1] for score in dice_scores_data_aug])
mean_dice_lv_data_aug = np.mean([score[2] for score in dice_scores_data_aug])
std_dice_lv_data_aug = np.std([score[2] for score in dice_scores_data_aug])

print(f'RV Dice Coefficient With Data Augmentation: Mean={mean_dice_rv_data_aug:.4f}, SD={std_dice_rv_data_aug:.4f}')
print(f'MYO Dice Coefficient With Data Augmentation: Mean={mean_dice_myo_data_aug:.4f}, SD={std_dice_myo_data_aug:.4f}')
print(f'LV Dice Coefficient With Data Augmentation: Mean={mean_dice_lv_data_aug:.4f}, SD={std_dice_lv_data_aug:.4f}')

# Save segmentation results for UNet with Data Augmentation
save_segmentation_results(net_data_aug, test_set, device, 'result', num_samples=3, model_name='baseline_unet_data_aug')


# --- UNet with Soft Dice Loss ---
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.sigmoid = nn.Sigmoid()


    def forward(self, pred_logits, targets_gt): # pred_logits (B, N_CLASS, H, W), targets_gt (B, 1, H, W) raw labels
        inputs_probs = self.sigmoid(pred_logits)  # Convert logits to probabilities (B, N_CLASS, H, W)
        targets_multilabel = convert_to_multi_labels(targets_gt)  # Convert GT to (B, N_CLASS, H, W) binary

        # Sum over spatial dimensions (H, W)
        intersection = (inputs_probs * targets_multilabel).sum(dim=(2, 3))
        union_sum = inputs_probs.sum(dim=(2, 3)) + targets_multilabel.sum(dim=(2, 3))

        dice_coefficient_per_class = (2. * intersection + self.smooth) / (union_sum + self.smooth)
        
        # Average Dice coefficient over classes and then over the batch
        dice_loss = 1 - dice_coefficient_per_class.mean() 
        return dice_loss

print("\n--- Training UNet with Soft Dice Loss (and Data Augmentation for training) ---")
net_soft_dice = UNet(n_channels=1, n_classes=3, C_base=32)
optimizer_soft_dice = torch.optim.Adam(net_soft_dice.parameters(), lr=0.001) # lr=0.001 as per user
lr_scheduler_soft_dice = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_soft_dice, gamma=0.95)

dice_loss_fn = SoftDiceLoss() # Renamed instance

solver_soft_dice = lab.Solver( # Renamed solver
    model=net_soft_dice,
    optimizer=optimizer_soft_dice,
    criterion=dice_loss_fn, # Use the SoftDiceLoss instance
    lr_scheduler=lr_scheduler_soft_dice,
)

# Training with augmented data as per user's setup for other models
solver_soft_dice.train(
    epochs=50, # Fewer for testing
    data_loader=dataloader_train_aug, # Using augmented training data
    val_loader=dataloader_val_aug,   # Using non-augmented validation data
    img_name='soft_dice_loss' # Model/run name
)

print("\n--- Evaluating UNet with Soft Dice Loss (Accuracy) ---")
accuracy_scores_soft_dice = []
net_soft_dice.to(device)
net_soft_dice.eval()

for images, labels_gt in dataloader_test: # Evaluate on original non-augmented test set
    images = images.to(device)  
    labels_gt = labels_gt.to(device) 
    with torch.no_grad():
        preds_logits = net_soft_dice(images)
    preds_binary = torch.sigmoid(preds_logits) > 0.5
    labels_multilabel = convert_to_multi_labels(labels_gt)

    accuracy_rv_soft_dice = get_accuracy(preds_binary[:, 0, :, :], labels_multilabel[:, 0, :, :])
    accuracy_myo_soft_dice = get_accuracy(preds_binary[:, 1, :, :], labels_multilabel[:, 1, :, :])
    accuracy_lv_soft_dice = get_accuracy(preds_binary[:, 2, :, :], labels_multilabel[:, 2, :, :])
    accuracy_scores_soft_dice.append((accuracy_rv_soft_dice, accuracy_myo_soft_dice, accuracy_lv_soft_dice))

mean_accuracy_rv_soft_dice = np.mean([score[0] for score in accuracy_scores_soft_dice])
std_accuracy_rv_soft_dice = np.std([score[0] for score in accuracy_scores_soft_dice])
mean_accuracy_myo_soft_dice = np.mean([score[1] for score in accuracy_scores_soft_dice])
std_accuracy_myo_soft_dice = np.std([score[1] for score in accuracy_scores_soft_dice])
mean_accuracy_lv_soft_dice  = np.mean([score[2] for score in accuracy_scores_soft_dice])
std_accuracy_lv_soft_dice  = np.std([score[2] for score in accuracy_scores_soft_dice])

print(f'RV Accuracy With Soft Dice Loss: Mean={mean_accuracy_rv_soft_dice:.4f}, SD={std_accuracy_rv_soft_dice:.4f}')
print(f'MYO Accuracy With Soft Dice Loss: Mean={mean_accuracy_myo_soft_dice:.4f}, SD={std_accuracy_myo_soft_dice:.4f}')
print(f'LV Accuracy With Soft Dice Loss: Mean={mean_accuracy_lv_soft_dice:.4f}, SD={std_accuracy_lv_soft_dice:.4f}')

# Save segmentation results for UNet with Soft Dice Loss
save_segmentation_results(net_soft_dice, test_set, device, 'result', num_samples=3, model_name='soft_dice_loss_unet') # Changed model_name for clarity

# --- Writing Results to File ---
# Ensure result directory exists for the text file
output_file_path = os.path.join('result', 'output_results.txt')
print(f"\n--- Writing all results to {output_file_path} ---")

with open(output_file_path, 'w') as file:
    file.write("--- Baseline UNet ---\n")
    file.write(f'RV Dice Coefficient: Mean={mean_dice_rv:.4f}, SD={std_dice_rv:.4f}\n')
    file.write(f'MYO Dice Coefficient: Mean={mean_dice_myo:.4f}, SD={std_dice_myo:.4f}\n')
    file.write(f'LV Dice Coefficient: Mean={mean_dice_lv:.4f}, SD={std_dice_lv:.4f}\n\n')

    file.write("--- UNet No Shortcut ---\n")
    file.write(f'RV Dice Coefficient Without Shortcut: Mean={mean_dice_rv_no_shortcut:.4f}, SD={std_dice_rv_no_shortcut:.4f}\n')
    file.write(f'MYO Dice Coefficient Without Shortcut: Mean={mean_dice_myo_no_shortcut:.4f}, SD={std_dice_myo_no_shortcut:.4f}\n')
    file.write(f'LV Dice Coefficient Without Shortcut: Mean={mean_dice_lv_no_shortcut:.4f}, SD={std_dice_lv_no_shortcut:.4f}\n\n')

    # Accuracy for baseline UNet (Data Aug section in original was for a new model)
    # The original code calculated accuracy for net_data_aug and net_soft_dice.
    # Let's assume the "RV Accuracy: Mean={mean_accuracy_rv}, SD={std_accuracy_rv}" 
    # in the original output file was a typo and meant for one of the later models,
    # or it was from an earlier run of the baseline where accuracy was also computed.
    # For now, I will only write metrics that were explicitly computed in this script flow.
    
    file.write("--- UNet with Data Augmentation ---\n")
    file.write(f'RV Accuracy (Data Aug): Mean={mean_accuracy_rv_aug:.4f}, SD={std_accuracy_rv_aug:.4f}\n')
    file.write(f'MYO Accuracy (Data Aug): Mean={mean_accuracy_myo_aug:.4f}, SD={std_accuracy_myo_aug:.4f}\n')
    file.write(f'LV Accuracy (Data Aug): Mean={mean_accuracy_lv_aug:.4f}, SD={std_accuracy_lv_aug:.4f}\n')
    file.write(f'RV Dice Coefficient With Data Augmentation: Mean={mean_dice_rv_data_aug:.4f}, SD={std_dice_rv_data_aug:.4f}\n')
    file.write(f'MYO Dice Coefficient With Data Augmentation: Mean={mean_dice_myo_data_aug:.4f}, SD={std_dice_myo_data_aug:.4f}\n')
    file.write(f'LV Dice Coefficient With Data Augmentation: Mean={mean_dice_lv_data_aug:.4f}, SD={std_dice_lv_data_aug:.4f}\n\n')
    
    file.write("--- UNet with Soft Dice Loss ---\n")
    file.write(f'RV Accuracy With Soft Dice Loss: Mean={mean_accuracy_rv_soft_dice:.4f}, SD={std_accuracy_rv_soft_dice:.4f}\n')
    file.write(f'MYO Accuracy With Soft Dice Loss: Mean={mean_accuracy_myo_soft_dice:.4f}, SD={std_accuracy_myo_soft_dice:.4f}\n')
    file.write(f'LV Accuracy With Soft Dice Loss: Mean={mean_accuracy_lv_soft_dice:.4f}, SD={std_accuracy_lv_soft_dice:.4f}\n')

print(f"All metrics saved to {output_file_path}")
print("Segmentation sample images saved in 'result/' subdirectories.")