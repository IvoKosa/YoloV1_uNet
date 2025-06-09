import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, num_classes=6, lambda_coord=5, lambda_noobj=0.5):
        """
        YOLO loss function.
        Parameters:
          S (int): Grid size.
          B (int): Number of bounding box predictions per grid cell.
          num_classes (int): Total number of detection classes.
          lambda_coord (float): Weight for coordinate loss.
          lambda_noobj (float): Weight for no-object confidence loss.
          
        Ground truth is assumed to be of shape (N, S, S, 5 + num_classes), where the 5
        corresponds to (x, y, w, h, conf).
        Predictions are assumed to be of shape (N, S, S, B*5 + num_classes), where the first
        B*5 values are for the B predicted boxes (each with 5 numbers) and the remaining are class scores.
        """
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def xywh_to_xyxy(self, box):
            cx, cy, w, h = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            return torch.stack([x1, y1, x2, y2], dim=-1)

    def forward(self, predictions, target):
        # predictions: [N, S, S, B*5 + num_classes]
        # target:      [N, S, S, 5 + num_classes]
        N = predictions.size(0)
        
        # Predicted boxes: reshape to [N, S, S, B, 5]
        pred_boxes = predictions[..., :self.B*5].view(N, self.S, self.S, self.B, 5)
        pred_class = predictions[..., self.B*5:]    # [N, S, S, num_classes]
        
        target_box = target[..., :5]                # [N, S, S, 5]
        target_class = target[..., 5:]              # [N, S, S, num_classes]
        
        obj_mask = target_box[..., 4] > 0           # [N, S, S]
        
        target_box_exp = target_box.unsqueeze(3).expand(N, self.S, self.S, self.B, 5)
        
        target_boxes_xyxy = self.xywh_to_xyxy(target_box_exp[..., :4])      # [N, S, S, B, 4]
        pred_boxes_xyxy = self.xywh_to_xyxy(pred_boxes[..., :4])            # [N, S, S, B, 4]
        
        inter_x1 = torch.max(pred_boxes_xyxy[..., 0], target_boxes_xyxy[..., 0])
        inter_y1 = torch.max(pred_boxes_xyxy[..., 1], target_boxes_xyxy[..., 1])
        inter_x2 = torch.min(pred_boxes_xyxy[..., 2], target_boxes_xyxy[..., 2])
        inter_y2 = torch.min(pred_boxes_xyxy[..., 3], target_boxes_xyxy[..., 3])
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        
        area_pred = ((pred_boxes_xyxy[..., 2] - pred_boxes_xyxy[..., 0]).clamp(min=0) *
                     (pred_boxes_xyxy[..., 3] - pred_boxes_xyxy[..., 1]).clamp(min=0))
        area_target = ((target_boxes_xyxy[..., 2] - target_boxes_xyxy[..., 0]).clamp(min=0) *
                       (target_boxes_xyxy[..., 3] - target_boxes_xyxy[..., 1]).clamp(min=0))
        union_area = area_pred + area_target - inter_area + 1e-6
        iou = inter_area / union_area                               # [N, S, S, B]
        
        iou_max, best_box = torch.max(iou, dim=-1)                  # best_box: [N, S, S]
        
        responsible_mask = torch.zeros_like(iou, dtype=torch.bool)  # [N, S, S, B]
        responsible_mask.scatter_(-1, best_box.unsqueeze(-1), True)
        
        obj_mask_exp = obj_mask.unsqueeze(-1).expand_as(iou)        # [N, S, S, B]
        
        # ---------------- Localisation Loss ----------------
        target_xy = target_box_exp[..., 0:2]                        # [N, S, S, B, 2]
        coord_loss = F.mse_loss(
            pred_boxes[..., 0:2][obj_mask_exp & responsible_mask],
            target_xy[obj_mask_exp & responsible_mask],
            reduction='sum'
        )
        
        pred_wh = torch.sqrt(torch.clamp(pred_boxes[..., 2:4], min=1e-6))
        target_wh = torch.sqrt(target_box_exp[..., 2:4] + 1e-6)
        size_loss = F.mse_loss(
            pred_wh[obj_mask_exp & responsible_mask],
            target_wh[obj_mask_exp & responsible_mask],
            reduction='sum'
        )
        
        # ---------------- Confidence Loss ----------------
        conf_loss_obj = F.mse_loss(
            pred_boxes[..., 4][obj_mask_exp & responsible_mask],
            target_box_exp[..., 4][obj_mask_exp & responsible_mask],
            reduction='sum'
        )
        
        noobj_mask = ~(obj_mask.unsqueeze(-1) & responsible_mask)
        conf_loss_noobj = F.mse_loss(
            pred_boxes[..., 4][noobj_mask],
            torch.zeros_like(pred_boxes[..., 4][noobj_mask]),
            reduction='sum'
        )
        
        # ---------------- Classification Loss ----------------
        class_loss = F.mse_loss(
            pred_class[obj_mask],
            target_class[obj_mask],
            reduction='sum'
        )
        
        total_loss = (self.lambda_coord * (coord_loss + size_loss) +
                      conf_loss_obj +
                      self.lambda_noobj * conf_loss_noobj +
                      class_loss)
        
        return total_loss / N
