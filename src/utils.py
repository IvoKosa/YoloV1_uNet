import cv2, argparse, yaml, torch, json
import numpy as np
from typing import Dict
from torchvision.ops import box_iou, nms

def bounding_boxifier(img: np.ndarray, CLASS_RGB_DICT: Dict[str, tuple], min_area: int = 50):
    """ Given a segmented image from the labeled dataset removes all unused classes and returns the image and corresponding bounding boxes

    :param img: OpenCV Image
    :param CLASS_RGB_DICT: Dict[str, tuple] dictionary containing the class labels (str) and a tuple of RGB values (R, G, B)
    :param min_area: Minimum area of pixel cluster to count as an object [Default = 50]

    :return masked: OpenCV Image with unused classes removed
    :return bounding_boxes: an array of tuples (x, y, w, h) denoting bounding box parameters
    """

    bounding_boxes = []
    mask_total = np.zeros(img.shape[:2], dtype=bool)
    for label, rgb in CLASS_RGB_DICT.items():
        if label == 'background':
            continue
        bgr = (rgb[2], rgb[1], rgb[0])
        mask_color = cv2.inRange(img, np.array(bgr), np.array(bgr))
        mask_total |= mask_color.astype(bool)
        contours, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                bounding_boxes.append((label, x, y, w, h))
    masked = np.zeros_like(img)
    masked[mask_total] = img[mask_total]
    return masked, bounding_boxes

def get_selective_transform(counts_json_path, base_transforms, extra_transforms, class_id_to_name, threshold=None):
    """
    Reads a counts.json file and returns a transform function that applies base_transforms
    to every sample and then (if any bounding box label in the sample belongs to a less used class)
    applies extra_transforms.

    Parameters:
      counts_json_path (str): Path to the counts.json file.
      base_transforms (callable): A transform function (e.g. Albumentations pipeline) that takes
                                  image, mask, bboxes, labels as input.
      extra_transforms (callable): A transform function that provides additional augmentation.
      class_id_to_name (dict): Mapping from numeric class IDs to class names.
                               For example: {1: "car", 2: "pedestrian", ...}
      threshold (float, optional): A threshold for image_count. Classes with image_count below this
                                   value will receive extra transforms. If None, the average
                                   image_count will be used.
    Returns:
      A transform function that can be passed as the transforms argument to your dataset.
    """
    with open(counts_json_path, "r") as f:
        counts = json.load(f)
    
    image_counts = [v["image_count"] for v in counts.values()]
    avg_count = sum(image_counts) / len(image_counts) if image_counts else 0
    threshold = threshold if threshold is not None else avg_count

    less_used_classes = [cls for cls, v in counts.items() if v["image_count"] < threshold]

    less_used_ids = set([cid for cid, cname in class_id_to_name.items() if cname in less_used_classes])

    def selective_transform(image, mask, bboxes, labels):
        """
        Applies base_transforms to every sample.
        Then, if any label in the sample belongs to a less used class, applies extra_transforms.
        """
        transformed = base_transforms(image=image, mask=mask, bboxes=bboxes, labels=labels)
        image_out = transformed["image"]
        mask_out = transformed["mask"]
        bboxes_out = transformed["bboxes"]
        labels_out = transformed["labels"]

        if any(label in less_used_ids for label in labels_out):
            extra = extra_transforms(image=image_out, mask=mask_out, bboxes=bboxes_out, labels=labels_out)
            image_out = extra["image"]
            mask_out = extra["mask"]
            bboxes_out = extra["bboxes"]
            labels_out = extra["labels"]

        return {"image": image_out, "mask": mask_out, "bboxes": bboxes_out, "labels": labels_out}

    return selective_transform

def config_dict_loader(param_name: str = None, config_dir: str = None):

    parser = argparse.ArgumentParser(description='Argument parser for config')

    if config_dir is not None:
        parser.add_argument('--config', dest='config_path',
                            default=config_dir, type=str)
    else:
        parser.add_argument('--config', dest='config_path',
                            default='/cs/student/projects1/rai/2024/ivokosa/object_detection/project_ivokosa/src/config.yaml', type=str)
    
    with open(parser.parse_args().config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    if param_name is not None:
        return config[param_name]
    
    return config

def convert_rgb_mask_to_label(mask, color_map):

    mask = np.transpose(mask.detach().cpu().numpy(), (0, 1, 2))
    h, w, _ = mask.shape
    label_mask = np.zeros((h, w), dtype=np.uint8)
    mapping = {}
    for i, (class_name, rgb) in enumerate(color_map.items()):
        mapping[class_name] = i
        match = np.all(mask == np.array(rgb, dtype=mask.dtype), axis=-1)
        label_mask[match] = i
    # for class_name, label in mapping.items():
    #     count = np.sum(label_mask == label)
    #     print(f"Class {class_name} (label {label}): {count} pixels")
    return torch.from_numpy(label_mask)

def normalize_image(image: torch.Tensor) -> torch.Tensor:
    image = image.float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
    normalized_image = (image - mean) / std
    return normalized_image

def create_detection_target(annotations, image_size=448, S=7, num_classes=6):
    """
    Constructs a YOLO detection target from an annotations dictionary.
    This version produces a target tensor of shape [S, S, num_classes+5].
    
    Parameters:
      annotations (dict): Dictionary with keys:
          'boxes': Tensor of shape [num_boxes, 4] containing bounding boxes in [x, y, w, h] format.
          'labels': Tensor of shape [num_boxes] containing integer class labels (excluding background).
      image_size (int): The size of the image (assumed square, e.g., 448).
      S (int): Grid size (e.g., 7).
      num_classes (int): Total number of classes (excluding background, e.g., 6).
      
    Returns:
      target: A tensor of shape [S, S, num_classes+5] that serves as the YOLO detection target.
    """
    target = np.zeros((S, S, num_classes + 5), dtype=np.float32)
    
    cell_size = image_size / S

    boxes = annotations['boxes']
    labels = annotations['labels']
    
    if hasattr(boxes, 'cpu'):
        boxes = boxes.cpu().numpy()
    else:
        boxes = np.array(boxes)
        
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()
    else:
        labels = np.array(labels)

    for i in range(boxes.shape[0]):
        cx, cy, w, h = boxes[i]

        col = int(cx // cell_size)
        row = int(cy // cell_size)

        col = np.clip(col, 0, S - 1)
        row = np.clip(row, 0, S - 1)

        cell_x = (cx - col * cell_size) / cell_size
        cell_y = (cy - row * cell_size) / cell_size

        norm_w = w / image_size
        norm_h = h / image_size

        bbox_vector = np.array([cell_x, cell_y, norm_w, norm_h, 1], dtype=np.float32)

        target[row, col, 0:5] = bbox_vector

        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[labels[i]] = 1

        target[row, col, 5:] = one_hot

    return torch.from_numpy(target)
