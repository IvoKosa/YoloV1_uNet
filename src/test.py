import utils
import torch
import torch as th
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from model import YOLO_UNet 
from dataloader import CamVidDataset

#############################################
#  Model Loading & Forward Pass Functions   #
#############################################

def load_full_model(model_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Loads the full model (UNet + YOLO head) from saved weights.
    """
    model = YOLO_UNet(num_seg_classes=6, num_det_classes=6, num_anchors=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def forward_pass(model, input_tensor):
    """
    Performs a forward pass using the full model.
    Returns segmentation mask output and YOLO detection predictions.
    """
    with torch.no_grad():
        seg_out, yolo_out = model(input_tensor)
    return seg_out, yolo_out

#############################################
#         Utility Display Functions         #
#############################################

def colorize_mask(mask, class_colors):
    """
    Converts a 2D mask (H, W) with class indices into a colored RGB image.
    """
    h, w = mask.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in class_colors.items():
        color_image[mask == class_idx] = color
    return color_image

def tensor_to_pil_image(image_tensor, is_normalized=False):
    """
    Converts an image tensor (shape: [3, H, W]) to a PIL image.
    """
    image_tensor = image_tensor.detach().cpu().clone()
    
    if is_normalized:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        image_tensor = image_tensor * std + mean
    else:
        if image_tensor.max() > 1:
            image_tensor = image_tensor / 255.0
    
    image_tensor = torch.clamp(image_tensor, 0, 1)
    return to_pil_image(image_tensor)

def draw_yolo_predictions(image, yolo_pred, image_size=448, S=7, B=2, threshold=0.5):
    """
    Draws YOLO predicted bounding boxes and class labels on the image.
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=12)
    except Exception:
        font = ImageFont.load_default()
    
    class_colors = {
        0: (128, 128, 128), 
        1: (255, 0, 0),    
        2: (0, 255, 0),   
        3: (0, 0, 255),    
        4: (255, 255, 0),    
        5: (0, 255, 255)     
    }
    class_names = {
        0: "bg",
        1: "car",
        2: "ped",
        3: "bike",
        4: "moto",
        5: "truck"
    }
    
    yolo_pred = yolo_pred[0].cpu().numpy()
    cell_size = image_size / S

    for i in range(S):
        for j in range(S):
            cell_pred = yolo_pred[i, j] 
            class_scores = cell_pred[B*5:]
            pred_class = np.argmax(class_scores)
            if pred_class == 0:
                continue 
            for b in range(B):
                base = b * 5
                bx, by, bw, bh, conf = cell_pred[base:base+5]
                if conf < threshold:
                    continue
                center_x = j * cell_size + bx * cell_size
                center_y = i * cell_size + by * cell_size
                box_w = bw * image_size
                box_h = bh * image_size
                x_min = center_x - box_w / 2
                y_min = center_y - box_h / 2
                x_max = center_x + box_w / 2
                y_max = center_y + box_h / 2
                color = class_colors.get(pred_class, (255, 255, 255))
                draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
                label_text = f"{class_names.get(pred_class, str(pred_class))} {conf:.2f}"
                draw.text((x_min, y_min), label_text, fill=color, font=font)
    return image

def draw_ground_truth_boxes(image, gt_boxes, unified_color=(0, 0, 0)):
    """
    Draws ground truth bounding boxes on the image using a unified color.
    """
    draw = ImageDraw.Draw(image)
    for box in gt_boxes:
        x, y, w, h = box
        x_min, y_min = x, y
        x_max, y_max = x + w, y + h
        draw.rectangle([x_min, y_min, x_max, y_max], outline=unified_color, width=2)
    return image

def overlay_segmentation_on_image(original_img, seg_mask, seg_class_colors, opacity=0.3):
    """
    Overlays segmentation mask (excluding background) on the original image with low opacity.
    
    Args:
        original_img (PIL.Image): The original image.
        seg_mask (np.array): 2D numpy array with predicted segmentation class indices.
        seg_class_colors (dict): Dictionary mapping class indices to RGB colors.
        opacity (float): The overlay opacity (0.0-1.0).
    
    Returns:
        PIL.Image: Image with segmentation overlay.
    """
    base = original_img.convert("RGBA")
    
    width, height = base.size
    seg_overlay = np.zeros((height, width, 4), dtype=np.uint8)
    
    for cls, color in seg_class_colors.items():
        if cls == 0:
            continue  
        indices = (seg_mask == cls)
        seg_overlay[indices, 0] = color[0]
        seg_overlay[indices, 1] = color[1]
        seg_overlay[indices, 2] = color[2]
        seg_overlay[indices, 3] = int(255 * opacity)
    
    overlay_img = Image.fromarray(seg_overlay, mode="RGBA")
    composite = Image.alpha_composite(base, overlay_img)
    return composite

def save_full_outputs(seg_output, yolo_pred, input_img, gt_mask, targets, save_prefix="output"):
    """
    Saves segmentation outputs, ground truth mask, and draws YOLO predictions and ground truth boxes on the input image.
    Also saves an image with the segmentation overlay (for non-background classes) on the original image.
    """
    seg_class_colors = {
        0: [0, 0, 0],  
        1: [255, 0, 0],   
        2: [0, 255, 0],     
        3: [0, 0, 255],    
        4: [255, 255, 0],  
        5: [0, 255, 255]    
    }
    
    pred_mask = seg_output.argmax(dim=1)[0].cpu().numpy()  
    seg_color = colorize_mask(pred_mask, seg_class_colors)
    seg_img = Image.fromarray(seg_color)
    seg_img.save(f"{save_prefix}_segmentation.png")
    
    gt_mask_np = gt_mask[0].cpu().numpy()
    gt_color = colorize_mask(gt_mask_np, seg_class_colors)
    gt_img = Image.fromarray(gt_color)
    gt_img.save(f"{save_prefix}_ground_truth.png")
    
    orig_img = tensor_to_pil_image(input_img[0])
    
    seg_overlay_img = overlay_segmentation_on_image(orig_img, pred_mask, seg_class_colors, opacity=0.3)
    seg_overlay_img.save(f"{save_prefix}_segmentation_overlay.png")
    
    img_with_boxes = orig_img.copy()
    img_with_boxes = draw_yolo_predictions(img_with_boxes, yolo_pred, image_size=input_img.shape[-1])
    
    if 'boxes' in targets:
        gt_boxes = targets['boxes'][0].cpu().numpy()
        img_with_boxes = draw_ground_truth_boxes(img_with_boxes, gt_boxes)
    
    img_with_boxes.save(f"{save_prefix}_detections.png")

#############################################
#         Evaluation Helper Functions       #
#############################################

def decode_yolo(yolo_pred, image_size=448, S=7, B=2, threshold=0.5):
    """
    Decodes YOLO output (for one image) into bounding boxes, confidence scores, and class labels.
    Returns:
        boxes: np.array of shape [num_boxes, 4] in (x, y, w, h) format.
        scores: np.array of shape [num_boxes] with confidence scores.
        classes: np.array of shape [num_boxes] with predicted class indices.
    """
    boxes = []
    scores = []
    classes = []
    cell_size = image_size / S
    for i in range(S):
        for j in range(S):
            cell_pred = yolo_pred[i, j]
            class_scores = cell_pred[B*5:]
            pred_class = np.argmax(class_scores)
            if pred_class == 0:
                continue
            for b in range(B):
                base = b * 5
                bx, by, bw, bh, conf = cell_pred[base:base+5]
                if conf < threshold:
                    continue
                center_x = j * cell_size + bx * cell_size
                center_y = i * cell_size + by * cell_size
                box_w = bw * image_size
                box_h = bh * image_size
                x_min = center_x - box_w / 2
                y_min = center_y - box_h / 2
                boxes.append([x_min, y_min, box_w, box_h])
                scores.append(conf)
                classes.append(pred_class)
    return np.array(boxes), np.array(scores), np.array(classes)

#############################################
#         Segmentation Metrics              #
#############################################

def compute_segmentation_iou(preds, gts, num_classes):
    """
    Compute mean Intersection-over-Union (mIoU) for segmentation predictions.
    """
    ious = []
    for pred, gt in zip(preds, gts):
        iou_per_class = []
        for cls in range(num_classes):
            pred_cls = (pred == cls)
            gt_cls = (gt == cls)
            intersection = np.logical_and(pred_cls, gt_cls).sum()
            union = np.logical_or(pred_cls, gt_cls).sum()
            if union == 0:
                continue
            iou_per_class.append(intersection / union)
        if iou_per_class:
            ious.append(np.mean(iou_per_class))
    if len(ious) == 0:
        return 0.0
    return np.mean(ious)

#############################################
#           YOLO Detection Metrics          #
#############################################

def convert_to_xyxy(box):
    """
    Converts a box from [x, y, w, h] to [x1, y1, x2, y2] format.
    """
    x, y, w, h = box
    return [x, y, x + w, y + h]

def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two boxes.
    Box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def average_precision(preds, gt_dict, iou_threshold=0.5):
    """
    Computes Average Precision (AP) for a single class.
    """
    preds = sorted(preds, key=lambda x: x[2], reverse=True)
    
    TP = np.zeros(len(preds))
    FP = np.zeros(len(preds))
    
    npos = sum([len(boxes) for boxes in gt_dict.values()])
    if npos == 0:
        return 0.0
    
    for i, (image_id, pred_box, score) in enumerate(preds):
        pred_box_xyxy = convert_to_xyxy(pred_box)
        gts = gt_dict.get(image_id, [])
        best_iou = 0
        best_gt_idx = -1
        
        for idx, (gt_box, detected) in enumerate(gts):
            gt_box_xyxy = convert_to_xyxy(gt_box)
            iou = compute_iou(pred_box_xyxy, gt_box_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        
        if best_iou >= iou_threshold:
            if not gts[best_gt_idx][1]:
                TP[i] = 1
                gts[best_gt_idx] = (gts[best_gt_idx][0], True)
            else:
                FP[i] = 1
        else:
            FP[i] = 1
    
    cum_TP = np.cumsum(TP)
    cum_FP = np.cumsum(FP)
    
    recalls = cum_TP / (npos + 1e-6)
    precisions = cum_TP / (cum_TP + cum_FP + 1e-6)
    
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    
    return ap

def mean_average_precision(predictions, ground_truths, iou_threshold=0.5):
    """
    Compute mean Average Precision (mAP) for object detection.
    """
    all_preds = {}
    all_gts = {}
    
    for image_id, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        pred_boxes, pred_scores, pred_classes = pred
        pred_boxes = [convert_to_xyxy(box) for box in pred_boxes]
        
        gt_boxes = gt['boxes']
        gt_boxes = [convert_to_xyxy(box) for box in gt_boxes]
        gt_classes = gt['labels']
        
        for box, score, cls in zip(pred_boxes, pred_scores, pred_classes):
            if cls not in all_preds:
                all_preds[cls] = []
            all_preds[cls].append((image_id, box, score))
        
        for box, cls in zip(gt_boxes, gt_classes):
            if cls not in all_gts:
                all_gts[cls] = {}
            if image_id not in all_gts[cls]:
                all_gts[cls][image_id] = []
            all_gts[cls][image_id].append((box, False))
    
    ap_list = []
    all_classes = set(all_preds.keys()).union(set(all_gts.keys()))
    for cls in all_classes:
        preds = all_preds.get(cls, [])
        gt_dict = all_gts.get(cls, {})
        ap = average_precision(preds, gt_dict, iou_threshold)
        ap_list.append(ap)
    
    mAP = np.mean(ap_list) if ap_list else 0.0
    return mAP

#############################################
#   Additional Metric Breakdown Functions   #
#############################################

def detailed_mean_average_precision(predictions, ground_truths, iou_threshold=0.5):
    """
    Computes per-class AP as well as overall mAP for object detection.
    Returns:
        mAP: Overall mean AP across classes.
        ap_dict: Dictionary mapping each class to its AP.
    """
    all_preds = {}
    all_gts = {}
    for image_id, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        pred_boxes, pred_scores, pred_classes = pred
        pred_boxes = [convert_to_xyxy(box) for box in pred_boxes]
        gt_boxes = gt['boxes']
        gt_boxes = [convert_to_xyxy(box) for box in gt_boxes]
        gt_classes = gt['labels']
        for box, score, cls in zip(pred_boxes, pred_scores, pred_classes):
            if cls not in all_preds:
                all_preds[cls] = []
            all_preds[cls].append((image_id, box, score))
        for box, cls in zip(gt_boxes, gt_classes):
            if cls not in all_gts:
                all_gts[cls] = {}
            if image_id not in all_gts[cls]:
                all_gts[cls][image_id] = []
            all_gts[cls][image_id].append((box, False))
    ap_dict = {}
    all_classes = set(all_preds.keys()).union(set(all_gts.keys()))
    for cls in all_classes:
        preds = all_preds.get(cls, [])
        gt_dict = all_gts.get(cls, {})
        ap = average_precision(preds, gt_dict, iou_threshold)
        ap_dict[cls] = ap
    mAP = np.mean(list(ap_dict.values())) if ap_dict else 0.0
    return mAP, ap_dict

def compute_segmentation_iou_per_class(preds, gts, num_classes):
    """
    Computes per-class IoU for segmentation predictions.
    Returns a dictionary mapping each class to its IoU.
    """
    total_intersections = np.zeros(num_classes)
    total_unions = np.zeros(num_classes)
    for pred, gt in zip(preds, gts):
        for cls in range(num_classes):
            pred_cls = (pred == cls)
            gt_cls = (gt == cls)
            intersection = np.logical_and(pred_cls, gt_cls).sum()
            union = np.logical_or(pred_cls, gt_cls).sum()
            total_intersections[cls] += intersection
            total_unions[cls] += union
    per_class_iou = {}
    for cls in range(num_classes):
        if total_unions[cls] > 0:
            per_class_iou[cls] = total_intersections[cls] / total_unions[cls]
        else:
            per_class_iou[cls] = None
    return per_class_iou

#############################################
#                Testing Loop               #
#############################################

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # model = load_full_model("project_ivokosa/results/cls_4_5_segement_95_95/full_model_weights_fin.pth", device=device)
    model = load_full_model("full_model_weights_fin.pth", device=device)
    split = 'train'

    training_config = utils.config_dict_loader(param_name='training_params')
    
    dataset = CamVidDataset(training_config['root_dir'], test_train_val=split)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    yolo_predictions = [] 
    yolo_ground_truths = [] 
    seg_predictions = []  
    seg_ground_truths = [] 

    model.eval()
    with torch.no_grad():
        for imgs, masks, targets in dataloader:
            input_tensor = imgs.to(device).float()
            gt_mask = masks.to(device)  
            seg_out, yolo_out = forward_pass(model, input_tensor)
            
            # --- Process YOLO predictions ---
            yolo_pred = yolo_out[0].cpu().numpy()
            boxes, scores, classes = decode_yolo(yolo_pred, image_size=input_tensor.shape[-1], S=7, B=2, threshold=0.5)

            yolo_predictions.append((boxes, scores, classes))
            gt_boxes = targets['boxes'][0].cpu().numpy() if 'boxes' in targets else np.empty((0,4))
            gt_classes = targets['labels'][0].cpu().numpy() if 'labels' in targets else np.empty((0,))
            yolo_ground_truths.append({'boxes': gt_boxes, 'labels': gt_classes})

            # --- Process segmentation predictions ---
            seg_pred = seg_out.argmax(dim=1)[0].cpu().numpy() 
            seg_predictions.append(seg_pred)
            seg_ground_truths.append(gt_mask[0].cpu().numpy())
    
    # Compute overall and per-class metrics for detection and segmentation
    mAP, per_class_AP = detailed_mean_average_precision(yolo_predictions, yolo_ground_truths, iou_threshold=0.5)
    mean_iou = compute_segmentation_iou(seg_predictions, seg_ground_truths, num_classes=6)
    per_class_iou = compute_segmentation_iou_per_class(seg_predictions, seg_ground_truths, num_classes=6)
    
    print("\nEvaluation Results:")
    print(f"Overall mAP for YOLO detection: {mAP:.4f}")
    print("Per-class mAP for YOLO detection:")
    for cls, ap in per_class_AP.items():
        print(f"{ap:.4f}")
    
    print(f"\nOverall Mean IoU for segmentation: {mean_iou:.4f}")
    print("Per-class IoU for segmentation:")
    for cls, iou in per_class_iou.items():
        if iou is not None:
            print(f"{iou:.4f}")
        else:
            print(f"Class {cls}: No ground truth")
    
    # --- Save sample outputs for visualization ---
    dataset = CamVidDataset(training_config['root_dir'], test_train_val=split)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    # img_id = 99
    # for i in range(img_id):
    for imgs, masks, targets in dataloader:
        input_tensor = imgs.to(device).float() 
        gt_mask = masks.to(device) 
        break 
    
    print(f'Img ID: {targets["img_path"]}')
    seg_out, yolo_out = forward_pass(model, input_tensor)
    save_full_outputs(seg_out, yolo_out, input_tensor, gt_mask, targets, save_prefix="camvid_sample")
