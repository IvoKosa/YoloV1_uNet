import os, utils, glob, cv2, json

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
from pathlib import Path
from PIL import Image

def CamVid_manager(preprocessor_params):
    """
    Downloads and automatically annotates the CamVid dataset. Assumes that this is being run as standard 
    from the src folder and CamVid dataset is present in the data folder with the following file structure:

    CamVid/
    ├─ test/
    ├─ test_labels/
    ├─ train/
    ├─ train_labels/
    ├─ val/
    ├─ val_labels/

    Obtained by running the following in python:

    import kagglehub
    
    kagglehub.dataset_download("carlolepelaars/camvid")
    """
    
    destination = os.path.join(Path(__file__).resolve().parent, '../data')
    camvid_folder = os.path.join(destination, "CamVid")
    anno_folder = os.path.join(destination, "annotations")

    for split in ['train', 'test', 'val']:
        target_path = os.path.join(anno_folder, split, "targets")
        mask_path = os.path.join(anno_folder, split, "masked_imgs")
        os.makedirs(target_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)
        
        counts = {cls: {"bbox_count": 0, "image_count": 0}
                  for cls in preprocessor_params['class_RGB_dict'] if cls != "background"}

        image_files = glob.glob(os.path.join(camvid_folder, (split + '_labels'), '*.png'))
        image_filenames = [os.path.splitext(os.path.basename(f))[0] for f in image_files]

        for idx, image in enumerate(image_files):
            img = cv2.imread(image)
            masked, bounding_boxes = utils.bounding_boxifier(img, preprocessor_params['class_RGB_dict'])
            cv2.imwrite(os.path.join(mask_path, (str(image_filenames[idx]) + '_masked.png')), masked)
            masked_np = np.array(masked)
            height, width = masked_np.shape[:2]

            categories = []
            cat2id = {}
            cat_id = 1
            for cls in preprocessor_params['class_RGB_dict']:
                if cls == 'background':
                    continue
                cat2id[cls] = cat_id
                categories.append({
                    "id": cat_id,
                    "name": cls,
                    "supercategory": "none"
                })
                cat_id += 1
            
            image_info = {
                "id": idx,
                "file_name": str(image_filenames[idx]),
                "width": width,
                "height": height
            }

            annotations = []
            ann_id = 1
            
            classes_in_image = set()
            for bbox in bounding_boxes:
                label, x, y, w, h = bbox
                if label == "background":
                    continue
                category_id = cat2id[label]
                area = int(w * h)
                ann = {
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": area,
                    "iscrowd": 0
                }
                annotations.append(ann)
                ann_id += 1

                counts[label]["bbox_count"] += 1
                classes_in_image.add(label)
            
            for label in classes_in_image:
                counts[label]["image_count"] += 1

            coco_output = {
                "images": [image_info],
                "annotations": annotations,
                "categories": categories
            }

            with open(os.path.join(target_path, (str(image_filenames[idx]) + '.json')), 'w') as f:
                json.dump(coco_output, f, indent=4)
        
        counts_file_path = os.path.join(anno_folder, split, "counts.json")
        with open(counts_file_path, "w") as cf:
            json.dump(counts, cf, indent=4)

class CamVidDataset(Dataset):
    """
    Main dataset class handling the CamVid dataset using COCO-style JSON annotations.
    """
    def __init__(self, root_dir: str, test_train_val: str = 'train', transforms=True, target_size=[448, 448]):
        super().__init__()
        self.root_dir = root_dir
        self.transforms = transforms
        self.test_train_val = test_train_val
        self.target_size = target_size
        self.config = utils.config_dict_loader(param_name='training_params')

        assert test_train_val in ['train', 'test', 'val'], "test_train_val must be 'train', 'test' or 'val'"
        self.img_dir = os.path.join(root_dir, 'CamVid', test_train_val)
        self.masked_img_dir = os.path.join(root_dir, 'annotations', test_train_val, 'masked_imgs')
        self.target_dir = os.path.join(root_dir, 'annotations', test_train_val, 'targets')

        self.pseudonyms = [filename[:-4] for filename in os.listdir(self.img_dir) if filename.endswith('.png')]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.CLASS_RGB_DICT = {
            'background': (0, 0, 0),
            'car': (64, 0, 128),
            'pedestrian': (64, 64, 0),
            'bicyclist': (0, 128, 192),
            'motorcycle/scooter': (192, 0, 192),
            'truck/bus': (192, 128, 192)
            }
        
        self.special_class_ids = self.config['low_instance_cls']

        if transforms:
            self.base_flip_prob = self.config['base_flip_prob']
            self.base_rot_prob = self.config['base_rot_prob']
        else:
            self.base_flip_prob = 0.0
            self.base_rot_prob = 0.0

    def __len__(self):
        return len(self.pseudonyms)
    
    def __getitem__(self, index):
        pid = self.pseudonyms[index]
        img_path = os.path.join(self.img_dir, f'{pid}.png')
        mask_path = os.path.join(self.masked_img_dir, f'{pid}_L_masked.png')
        json_path = os.path.join(self.target_dir, f'{pid}_L.json')

        img = Image.open(img_path).convert("RGB") 
        mask = Image.open(mask_path) 

        img = np.array(img) 
        mask = np.array(mask) 

        with open(json_path, 'r') as f:
            ann_data = json.load(f)

        annotations = ann_data.get("annotations", [])
        boxes_list = [] 
        labels_list = []
        areas = []

        for ann in annotations:
            boxes_list.append(ann["bbox"])
            labels_list.append(ann["category_id"])
            areas.append(ann["area"])

        flip = self.base_flip_prob
        rot = self.base_rot_prob

        if self.transforms and any(label in self.special_class_ids for label in labels_list):
            flip = self.config['flip_prob']
            rot = self.config['rot_prob']

        if self.test_train_val != 'train':
            flip = 0.0
            rot = 0.0

        transform = A.Compose([
            A.Resize(height=448, width=448),
            A.HorizontalFlip(p=flip),
            A.Rotate(limit=15, p=rot)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

        # transform = A.Compose([
        #     A.Resize(height=448, width=448),
        #     A.Rotate(limit=15, p=rot)
        # ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

        transformed = transform(image=img, mask=mask, bboxes=boxes_list, labels=labels_list)
        transformed_img = transformed['image']
        transformed_mask = transformed['mask']
        transformed_boxes = transformed['bboxes']
        transformed_labels = transformed['labels']

        to_tensor = A.Compose([ToTensorV2()])
        transformed = to_tensor(image=transformed_img, mask=transformed_mask)
        transformed_img = transformed['image']
        transformed_mask = transformed['mask']

        new_areas = [box[2] * box[3] for box in transformed_boxes]

        boxes_tensor = torch.as_tensor(transformed_boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(transformed_labels, dtype=torch.int64)
        areas_tensor = torch.as_tensor(new_areas, dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([int(pid)] if pid.isdigit() else [index]),
            "area": areas_tensor,
            "img_path": img_path
        }

        return transformed_img, utils.convert_rgb_mask_to_label(transformed_mask, self.CLASS_RGB_DICT), target
    
if __name__ == '__main__':

    CamVid_manager(utils.config_dict_loader('preprocessor_params'))

    # data = CamVidDataset('/cs/student/projects1/rai/2024/ivokosa/object_detection/project_ivokosa/data')
    # img, mask, tgt = data[0]



