import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np
import os
import matplotlib.pyplot as plt # Still needed for unet_save_segmentation_results

# --- Import the provided Solver ---
# Assuming the provided solver code is saved in a file named "bme1301_solver.py"
# in the same directory or a directory in PYTHONPATH.
# If it's directly in bme1312.lab2, then the original import is fine.
# For this example, let's assume it's bme1301_solver.py
try:
    from bme1312.lab2 import Solver # Use the provided Solver
except ImportError:
    print("ERROR: The provided 'bme1301_solver.py' file was not found.")
    print("Please ensure it's in the same directory or your PYTHONPATH.")
    # Define a dummy Solver to allow the rest of the script to be parsed
    class Solver:
        def __init__(self, *args, **kwargs): pass
        def train(self, *args, **kwargs): print("Dummy Solver: train called"); return {} # Return dummy history
        def validate(self, *args, **kwargs): print("Dummy Solver: validate called"); return 0.0
        def to_device(self, x): return x # Minimal passthrough

from bme1312.evaluation import get_DC # Keep this for our custom evaluation

from unet import (
    UNet,
    process_data,
    convert_to_multi_labels,
    save_segmentation_results as unet_save_segmentation_results
)
try:
    from attention_unet import AttentionUNet
    ATTENTION_UNET_AVAILABLE = True
except ImportError:
    print("Warning: attention_unet.py not found or AttentionUNet class not defined. Skipping AttentionUNet.")
    ATTENTION_UNET_AVAILABLE = False
    class AttentionUNet(nn.Module): # Placeholder
        def __init__(self, n_channels, n_classes, bilinear=True, C_base=32):
            super(AttentionUNet, self).__init__()
            self.passthrough = nn.Conv2d(n_channels, n_classes, kernel_size=1)
            self.n_classes = n_classes
        def forward(self, x): return self.passthrough(x)

from torch.utils.data import TensorDataset, DataLoader

# --- Helper: compute_distance_maps_for_hybrid_loss (same as before) ---
def compute_distance_maps_for_hybrid_loss(targets_batch_multilabel, device):
    B, N_C, H, W = targets_batch_multilabel.shape
    dist_map_fg_to_boundary_all_classes = torch.zeros_like(targets_batch_multilabel, dtype=torch.float32)
    dist_map_bg_to_boundary_all_classes = torch.zeros_like(targets_batch_multilabel, dtype=torch.float32)
    for b in range(B):
        for c in range(N_C):
            target_class_c_np = targets_batch_multilabel[b, c].cpu().numpy().astype(np.uint8)
            if np.any(target_class_c_np):
                dist_map_fg_to_boundary_all_classes[b, c] = torch.from_numpy(
                    distance_transform_edt(target_class_c_np)).float()
                dist_map_bg_to_boundary_all_classes[b, c] = torch.from_numpy(
                    distance_transform_edt(1 - target_class_c_np)).float()
            else:
                dist_map_fg_to_boundary_all_classes[b, c] = torch.zeros((H,W), dtype=torch.float32)
                dist_map_bg_to_boundary_all_classes[b, c] = torch.from_numpy(
                     distance_transform_edt(np.ones_like(target_class_c_np))).float()
    return (dist_map_fg_to_boundary_all_classes.to(device),
            dist_map_bg_to_boundary_all_classes.to(device))

# --- HybridLoss class (same as before) ---
class HybridLoss(nn.Module):
    def __init__(self, n_classes=3, smooth=1e-6, boundary_loss_alpha=1.0):
        super(HybridLoss, self).__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        self.boundary_loss_alpha = boundary_loss_alpha
        self.log_sigma_dice = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_ce = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_boundary = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_hausdorff = nn.Parameter(torch.tensor(0.0))

    def forward(self, preds_logits, targets_raw): # targets_raw are (B,1,H,W)
        B, N_C, H, W = preds_logits.shape
        current_device = preds_logits.device
        preds_probs = torch.sigmoid(preds_logits)
        targets_multilabel = convert_to_multi_labels(targets_raw.to(current_device))
        dist_map_fg_to_boundary, dist_map_bg_to_boundary = \
            compute_distance_maps_for_hybrid_loss(targets_multilabel.detach(), current_device)

        intersection = (preds_probs * targets_multilabel).sum(dim=(2, 3))
        pred_sum = preds_probs.sum(dim=(2, 3))
        target_sum = targets_multilabel.sum(dim=(2, 3))
        dice_coeff_per_batch_class = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        dice_loss = 1. - dice_coeff_per_batch_class.mean()

        ce_loss = F.binary_cross_entropy_with_logits(preds_logits, targets_multilabel, reduction='mean')
        
        min_dist_to_boundary = torch.min(dist_map_fg_to_boundary, dist_map_bg_to_boundary)
        boundary_region_weights = torch.exp(-self.boundary_loss_alpha * min_dist_to_boundary)
        boundary_loss_pixelwise = boundary_region_weights * (preds_probs - targets_multilabel).abs()
        boundary_loss = boundary_loss_pixelwise.mean()

        term1_hausdorff = preds_probs * dist_map_bg_to_boundary
        term2_hausdorff = (1. - preds_probs) * dist_map_fg_to_boundary
        hausdorff_loss = (term1_hausdorff.mean() + term2_hausdorff.mean()) / 2.0

        total_loss = (
            torch.exp(-self.log_sigma_dice) * dice_loss + self.log_sigma_dice +
            torch.exp(-self.log_sigma_ce) * ce_loss + self.log_sigma_ce +
            torch.exp(-self.log_sigma_boundary) * boundary_loss + self.log_sigma_boundary +
            torch.exp(-self.log_sigma_hausdorff) * hausdorff_loss + self.log_sigma_hausdorff
        )
        return total_loss

# --- New Solver for Hybrid Loss Task ---
class HybridLossTaskSolver(Solver): # Inherit from the provided Solver
    def _step(self, batch, is_compute_metrics=True) -> dict:
        image, seg_gt_raw = batch # seg_gt_raw is (B, 1, H, W) with original labels

        image = self.to_device(image)
        seg_gt_raw = self.to_device(seg_gt_raw)
        B, C_in, H, W = image.shape # C_in is input channels (e.g. 1)
        
        # Model prediction
        pred_seg_logits = self.model(image) # Output (B, N_CLASSES, H, W)
        
        # Calculate loss using the HybridLoss criterion
        # HybridLoss.forward expects (preds_logits, targets_raw)
        loss = self.criterion(pred_seg_logits, seg_gt_raw) 
        
        step_dict = {'loss': loss, 'batch_size': B}

        if not self.model.training and is_compute_metrics:
            pred_seg_probs = torch.sigmoid(pred_seg_logits) # (B, N_CLASSES, H, W)
            
            # Convert ground truth to multi-label for metric calculation
            # seg_gt_raw is already on device
            seg_gt_multilabel = convert_to_multi_labels(seg_gt_raw) # (B, N_CLASSES, H, W)
            
            n_model_classes = pred_seg_logits.shape[1] # Should be self.model.n_classes if available
            
            dice_scores_per_class_in_batch = []
            class_names = ['RV', 'MYO', 'LV'] # Assuming 3 classes

            for c_idx in range(n_model_classes):
                pred_binary_c = (pred_seg_probs[:, c_idx] > 0.5).float() # (B, H, W)
                target_binary_c = seg_gt_multilabel[:, c_idx].float()    # (B, H, W)
                
                # get_DC expects (H,W) or (B,H,W) and returns a scalar or tensor of Dice scores for the batch
                dc_c_batch = get_DC(pred_binary_c, target_binary_c)
                
                # If get_DC returns a tensor of per-image DCs, take the mean for this class over the batch
                # If it returns a scalar (already meaned), use it directly.
                # Assuming get_DC returns a scalar mean over the batch for this class.
                step_dict[f'metric_Dice_{class_names[c_idx]}'] = dc_c_batch.item() if torch.is_tensor(dc_c_batch) else float(dc_c_batch)
                dice_scores_per_class_in_batch.append(dc_c_batch.item() if torch.is_tensor(dc_c_batch) else float(dc_c_batch))
            
            if dice_scores_per_class_in_batch:
                step_dict['metric_avg_DiceCoefficient'] = np.mean(dice_scores_per_class_in_batch)
            else:
                step_dict['metric_avg_DiceCoefficient'] = 0.0
        return step_dict

    def visualize(self, data_loader, idx, *, dpi=100):
        # Similar to Lab2Solver.visualize but adapted for multi-class
        # This method is called by user, not automatically by train/validate
        import itertools # For islice
        from bme1301_solver import image_mask_overlay, imsshow # Assuming these are in the solver utils

        with torch.no_grad():
            if idx < 0 or idx >= len(data_loader.dataset): # Check against dataset length
                raise RuntimeError(f"idx {idx} is out of range for dataset size {len(data_loader.dataset)}.")

            # Fetch a single data point
            image_single, seg_gt_raw_single = data_loader.dataset[idx] # (C,H,W), (C,H,W)
            
            # Add batch dimension and move to device
            image_batch = image_single.unsqueeze(0).to(self.device) # (1, C_in, H, W)
            seg_gt_raw_batch = seg_gt_raw_single.unsqueeze(0).to(self.device) # (1, 1, H, W)

            self.model.eval()
            pred_seg_logits = self.model(image_batch)  # (1, N_CLASSES, H, W)
            pred_seg_probs = torch.sigmoid(pred_seg_logits) # (1, N_CLASSES, H, W)
            
            # For overall Dice, convert GT to multilabel and average per-class Dice
            seg_gt_multilabel = convert_to_multi_labels(seg_gt_raw_batch) # (1, N_CLASSES, H, W)
            
            avg_dc_sample = 0
            num_classes_for_dc = pred_seg_probs.shape[1]
            class_dcs = []
            for c_idx in range(num_classes_for_dc):
                pred_mask_c = (pred_seg_probs[0, c_idx] > 0.5).float() # (H,W)
                gt_mask_c = seg_gt_multilabel[0, c_idx].float() # (H,W)
                dc_val = get_DC(pred_mask_c, gt_mask_c)
                class_dcs.append(dc_val.item() if torch.is_tensor(dc_val) else float(dc_val))
            if class_dcs:
                avg_dc_sample = np.mean(class_dcs)

            # Prepare for plotting (move to CPU, convert to NumPy)
            image_np = self.to_numpy(image_single.squeeze()) # (H, W)
            
            # Ground truth: For visualization, we might want to show the original single-channel label
            # or a composite of multi-label. Let's show individual class GTs and Preds.
            class_names = ['RV', 'MYO', 'LV']

            num_plot_rows = num_classes_for_dc
            fig, axes = plt.subplots(num_plot_rows, 3, figsize=(12, 4 * num_plot_rows), dpi=dpi)
            if num_plot_rows == 1: axes = axes.reshape(1,-1) # Ensure axes is 2D

            fig.suptitle(f"Sample {idx} - Overall Avg DICE {avg_dc_sample:.3f}", fontsize=16)

            for c_idx in range(num_plot_rows):
                gt_mask_c_np = self.to_numpy(seg_gt_multilabel[0, c_idx])
                pred_mask_c_np = self.to_numpy(pred_seg_probs[0, c_idx] > 0.5)
                
                # Input Image (repeated for context)
                ax_img = axes[c_idx, 0]
                ax_img.imshow(image_np, cmap='gray')
                ax_img.set_title(f"Input Image (Context for {class_names[c_idx]})")
                ax_img.axis('off')

                # Ground Truth Mask
                ax_gt = axes[c_idx, 1]
                ax_gt.imshow(gt_mask_c_np, cmap='viridis', vmin=0, vmax=1) # Or use specific cmap for GT
                ax_gt.set_title(f"GT - {class_names[c_idx]} (DC: {class_dcs[c_idx]:.3f})")
                ax_gt.axis('off')

                # Predicted Mask
                ax_pred = axes[c_idx, 2]
                ax_pred.imshow(pred_mask_c_np, cmap='viridis', vmin=0, vmax=1)
                ax_pred.set_title(f"Pred - {class_names[c_idx]}")
                ax_pred.axis('off')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    BASE_OUTPUT_DIR = 'results_all_models_new_solver'
    LEARNING_RATE = 1e-4 
    BATCH_SIZE = 8 
    EPOCHS = 50 # For testing, use fewer like 5-10. For real training, 50-200.

    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)
    # The solver will create 'result/' subdir if img_name doesn't have it.
    # Let's ensure 'result/' exists for solver's default plot saving.
    if not os.path.exists('result'):
        os.makedirs('result')


    inputs, labels = process_data()
    inputs = inputs.unsqueeze(1)
    labels = labels.unsqueeze(1)
    print(f"Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")

    dataset = TensorDataset(inputs, labels)
    # --- Data Splitting (same robust logic as before) ---
    total_len = len(dataset)
    if total_len < 3: raise ValueError("Dataset too small for any split.")
    train_size = int(4/7 * total_len) if total_len >=7 else max(1, int(0.6 * total_len))
    val_size = int(1/7 * total_len) if total_len >=7 else max(1, int(0.2 * total_len))
    test_size = total_len - train_size - val_size
    if test_size <= 0 and total_len > train_size + val_size : test_size = total_len - train_size - val_size
    elif test_size <= 0 : val_size = total_len - train_size; test_size = 0
    if train_size == 0: train_size=1; val_size=0; test_size=0 # Ensure train is not 0
    
    # Ensure sum matches total_len, prioritize train, then val
    current_sum = train_size + val_size + test_size
    if current_sum != total_len:
        if current_sum < total_len: # Add remainder to train or test
            if test_size > 0 : test_size += (total_len - current_sum)
            else: train_size += (total_len - current_sum)
        else: # current_sum > total_len, reduce from test, then val
            if test_size > 0: test_size -= (current_sum - total_len)
            if test_size < 0: val_size += test_size; test_size = 0 # Add negative test_size to val_size
            if val_size < 0: train_size += val_size; val_size = 0 # Add negative val_size to train_size

    print(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")

    torch.manual_seed(42)
    if train_size > 0 and val_size > 0 and (test_size > 0 or test_size ==0) : # test can be 0
        # Ensure split list matches total_len
        split_lengths = []
        if train_size > 0: split_lengths.append(train_size)
        if val_size > 0: split_lengths.append(val_size)
        if test_size > 0: split_lengths.append(test_size)
        
        if sum(split_lengths) != total_len: # Adjust last element if sum is off
            if split_lengths:
                 split_lengths[-1] += (total_len - sum(split_lengths))

        if len(split_lengths) == 3:
            train_set, val_set, test_set = torch.utils.data.random_split(dataset, split_lengths)
        elif len(split_lengths) == 2: # train and val, no test
            train_set, val_set = torch.utils.data.random_split(dataset, split_lengths)
            test_set = val_set # Use val for test if test_size was 0
            print("Warning: Test set size is 0. Using validation set for testing.")
        elif len(split_lengths) == 1: # only train
            train_set = dataset
            val_set = dataset
            test_set = dataset
            print("Warning: Validation and Test set sizes are 0. Using training set for all.")
        else: # Should not happen
            raise ValueError("Problem with dataset split sizes.")
    else:
        raise ValueError("Train size is 0 or invalid split. Cannot proceed.")


    dataloader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dataloader_val = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    dataloader_test = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


    models_to_run = []
    models_to_run.append({'name': 'UNet_HybridLoss_NewSolver', 'class': UNet, 'C_base': 32})
    if ATTENTION_UNET_AVAILABLE:
        models_to_run.append({'name': 'AttUNet_HybridLoss_NewSolver', 'class': AttentionUNet, 'C_base': 32})

    all_results_summary = {}

    for model_config in models_to_run:
        model_name_str = model_config['name']
        ModelClass = model_config['class']
        c_base = model_config['C_base']
        
        model_output_dir = os.path.join(BASE_OUTPUT_DIR, model_name_str)
        if not os.path.exists(model_output_dir): os.makedirs(model_output_dir)

        print(f"\n--- Processing Model: {model_name_str} ---")
        net = ModelClass(n_channels=1, n_classes=3, C_base=c_base, bilinear=True).to(device)
        loss_fn = HybridLoss(n_classes=net.n_classes, boundary_loss_alpha=1.0).to(device)
        optimizer = torch.optim.Adam(list(net.parameters()) + list(loss_fn.parameters()), lr=LEARNING_RATE)
        
        # Use the new HybridLossTaskSolver
        solver = HybridLossTaskSolver(model=net, optimizer=optimizer, criterion=loss_fn, device=device)

        print(f"--- Training {model_name_str} ---")
        # The solver's train method will plot and save loss curves to result/<model_name_str>.png
        solver.train(
            epochs=EPOCHS,
            data_loader=dataloader_train,
            val_loader=dataloader_val,
            img_name=model_name_str # Used by solver for saving plot
        )
        print(f"\n--- Training Complete for {model_name_str} ---")
        
        # The solver's validate() method prints metrics to console but doesn't return a history dict.
        # We need a separate loop for final test evaluation if we want to collect scores.
        print(f"\n--- Evaluating {model_name_str} on Test Set ---")
        dice_scores_list = []
        net.eval()
        for images_test, labels_gt_test in dataloader_test:
            if not images_test.numel(): continue
            images_test = images_test.to(device)
            labels_gt_test = labels_gt_test.to(device) # raw labels
            with torch.no_grad():
                preds_logits_test = net(images_test) # (B, N_Class, H, W)
            
            preds_probs_test = torch.sigmoid(preds_logits_test)
            labels_multilabel_test = convert_to_multi_labels(labels_gt_test)

            if preds_probs_test.numel() > 0 and labels_multilabel_test.numel() > 0:
                batch_class_dcs = []
                for c_idx in range(preds_probs_test.shape[1]): # Iterate over classes
                    pred_c_binary = (preds_probs_test[:, c_idx] > 0.5).float()
                    label_c_binary = labels_multilabel_test[:, c_idx].float()
                    dc_val = get_DC(pred_c_binary, label_c_binary)
                    batch_class_dcs.append(dc_val.item() if torch.is_tensor(dc_val) else float(dc_val))
                
                # Store per-class DC for the batch (RV, MYO, LV)
                if len(batch_class_dcs) == 3: # Assuming 3 classes
                     dice_scores_list.append(tuple(batch_class_dcs))
                elif batch_class_dcs : # if some classes were calculated
                     # Pad with zeros if not all 3 classes were present/calculated (should not happen with fixed n_classes=3)
                     padded_dcs = batch_class_dcs + [0.0] * (3 - len(batch_class_dcs))
                     dice_scores_list.append(tuple(padded_dcs[:3]))


        model_metrics = {}
        if dice_scores_list:
            model_metrics['RV_Dice_Mean'] = np.mean([s[0] for s in dice_scores_list])
            model_metrics['RV_Dice_Std'] = np.std([s[0] for s in dice_scores_list])
            model_metrics['MYO_Dice_Mean'] = np.mean([s[1] for s in dice_scores_list])
            model_metrics['MYO_Dice_Std'] = np.std([s[1] for s in dice_scores_list])
            model_metrics['LV_Dice_Mean'] = np.mean([s[2] for s in dice_scores_list])
            model_metrics['LV_Dice_Std'] = np.std([s[2] for s in dice_scores_list])
        else:
            print(f"No dice scores recorded for {model_name_str} on test set. Setting metrics to 0.")
            model_metrics.update({k: 0 for k in ['RV_Dice_Mean', 'RV_Dice_Std', 'MYO_Dice_Mean', 'MYO_Dice_Std', 'LV_Dice_Mean', 'LV_Dice_Std']})
        
        all_results_summary[model_name_str] = model_metrics

        print(f"RV Dice ({model_name_str}): Mean={model_metrics['RV_Dice_Mean']:.4f}, SD={model_metrics['RV_Dice_Std']:.4f}")
        print(f"MYO Dice ({model_name_str}): Mean={model_metrics['MYO_Dice_Mean']:.4f}, SD={model_metrics['MYO_Dice_Std']:.4f}")
        print(f"LV Dice ({model_name_str}): Mean={model_metrics['LV_Dice_Mean']:.4f}, SD={model_metrics['LV_Dice_Std']:.4f}")

        if len(test_set) > 0:
            print(f"\n--- Saving Segmentation Samples for {model_name_str} using unet_save_segmentation_results ---")
            unet_save_segmentation_results(
                model=net,
                dataset_to_sample_from=test_set, # The Dataset object
                device=device,
                output_base_dir=model_output_dir,
                num_samples=min(3, len(test_set)),
                model_name="segmentation_samples_custom_func"
            )
            # Example of using the solver's visualize method for one sample
            if len(dataloader_test.dataset) > 0:
                 print(f"\n--- Visualizing one sample for {model_name_str} using solver.visualize ---")
                 try:
                    solver.visualize(dataloader_test, idx=0, dpi=100) # Visualize the first sample from test_loader
                 except Exception as e:
                    print(f"Could not run solver.visualize: {e}")
        else:
            print(f"Test set is empty for {model_name_str}. Skipping saving/visualizing samples.")


        output_file_path = os.path.join(model_output_dir, f'{model_name_str}_results.txt')
        print(f"\n--- Writing {model_name_str} results to {output_file_path} ---")
        with open(output_file_path, 'w') as file:
            file.write(f"--- Results for {model_name_str} ---\n")
            file.write(f"RV Dice Coefficient: Mean={model_metrics['RV_Dice_Mean']:.4f}, SD={model_metrics['RV_Dice_Std']:.4f}\n")
            file.write(f"MYO Dice Coefficient: Mean={model_metrics['MYO_Dice_Mean']:.4f}, SD={model_metrics['MYO_Dice_Std']:.4f}\n")
            file.write(f"LV Dice Coefficient: Mean={model_metrics['LV_Dice_Mean']:.4f}, SD={model_metrics['LV_Dice_Std']:.4f}\n\n")
        print(f"{model_name_str} metrics saved to {output_file_path}")

    summary_file_path = os.path.join(BASE_OUTPUT_DIR, 'all_models_summary_results.txt')
    print(f"\n--- Writing Overall Summary to {summary_file_path} ---")
    with open(summary_file_path, 'w') as file:
        file.write("--- Overall Model Comparison (Mean Dice Scores) ---\n")
        for model_name_key, metrics in all_results_summary.items():
            file.write(f"\n-- Model: {model_name_key} --\n")
            file.write(f"  RV Dice: Mean={metrics.get('RV_Dice_Mean', 0):.4f}, SD={metrics.get('RV_Dice_Std', 0):.4f}\n")
            file.write(f"  MYO Dice: Mean={metrics.get('MYO_Dice_Mean', 0):.4f}, SD={metrics.get('MYO_Dice_Std', 0):.4f}\n")
            file.write(f"  LV Dice: Mean={metrics.get('LV_Dice_Mean', 0):.4f}, SD={metrics.get('LV_Dice_Std', 0):.4f}\n")
    print(f"Overall summary saved to {summary_file_path}")
    print("\n--- All Processing Complete ---")