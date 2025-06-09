import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from model import YOLO_UNet  
from dataloader import CamVidDataset 
from loss import YOLOLoss

class Trainer():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.training_config = utils.config_dict_loader(param_name='training_params')

    def collate_function(self, batch):
        imgs, masks, targets = zip(*batch)
    
        imgs = torch.stack(imgs, dim=0)
        masks = torch.stack(masks, dim=0)
        
        detection_targets = [utils.create_detection_target(target) for target in targets]
        detection_targets = torch.stack(detection_targets, dim=0)
        
        return imgs, masks, detection_targets
    
    def validate(self, model, validation_loader, seg_criterion, yolo_loss_fn):
        model.eval() 
        total_seg_loss = 0.0 
        total_det_loss = 0.0
        total_val_loss = 0.0
        total = 0
        
        with torch.no_grad():
            for i, batch in enumerate(validation_loader):
                imgs, seg_masks, detection_targets = batch
                imgs = imgs.to(self.device).float()  
                seg_masks = seg_masks.to(self.device).long()
                detection_targets = detection_targets.to(self.device).float()

                seg_out, yolo_out = model(imgs)

                seg_loss = seg_criterion(seg_out, seg_masks)
                det_loss = yolo_loss_fn(yolo_out, detection_targets)

                total_seg_loss += seg_loss
                total_det_loss += det_loss
                # total_val_loss += val_loss
        total = len(validation_loader)
        total_seg_loss = total_seg_loss / total
        total_det_loss = total_det_loss / total
        val_loss = self.training_config['segmentation_loss_weight'] * total_seg_loss + self.training_config['detection_loss_weight'] * total_det_loss
        
        return total_seg_loss, total_det_loss , val_loss
    
    def train_full_model(self):
        """
        Trains the full model which outputs both segmentation masks and YOLO detections.
        
        Assumes the dataset returns a tuple:
        (img, seg_mask, detection_target)
        where:
        - img: [3, H, W] float tensor (normalized)
        - seg_mask: [H, W] long tensor with segmentation class labels
        - detection_target: [S, S, (B*5 + num_detection_classes)] float tensor representing YOLO ground truth
        """

        dataset = CamVidDataset(self.training_config['root_dir'], 
                                transforms=self.training_config['transforms'])
        
        dataloader = DataLoader(dataset, 
                                shuffle=self.training_config['shuffle'],
                                batch_size=self.training_config['batch_size'],
                                num_workers=self.training_config['num_workers'],
                                collate_fn=self.collate_function)
        
        val_dataset = CamVidDataset(self.training_config['root_dir'], 
                                transforms=self.training_config['transforms'], test_train_val='val')
        
        val_dataloader = DataLoader(val_dataset, 
                                shuffle=self.training_config['shuffle'],
                                batch_size=self.training_config['batch_size'],
                                num_workers=self.training_config['num_workers'],
                                collate_fn=self.collate_function)

        yolo_loss_fn = YOLOLoss()
        
        model = YOLO_UNet(num_seg_classes=6, num_det_classes=6, num_anchors=2)
        model.to(self.device)
        
        # weight=torch.tensor([1.0, 1.0, 1.0, 3.0, 3.0, 0.5]).to(self.device)
        seg_criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(model.parameters(), lr=self.training_config['learning_rate'])

        improvement = 0
        min_val_loss = float('inf')
        min_loss = float('inf')
        
        for epoch in range(self.training_config['epochs']):
            model.train()
            running_loss = 0.0
            running_seg_loss = 0.0
            running_det_loss = 0.0
            
            for i, batch in enumerate(dataloader):
                imgs, seg_masks, detection_targets = batch
                imgs = imgs.to(self.device).float()  
                seg_masks = seg_masks.to(self.device).long()
                detection_targets = detection_targets.to(self.device).float()
                optimizer.zero_grad()
                
                seg_out, yolo_out = model(imgs)

                seg_loss = seg_criterion(seg_out, seg_masks)
                
                det_loss = yolo_loss_fn(yolo_out, detection_targets)
                
                loss = self.training_config['segmentation_loss_weight'] * seg_loss + self.training_config['detection_loss_weight'] * det_loss
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                running_seg_loss += seg_loss.item()
                running_det_loss += det_loss.item()
                
                if i % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.training_config['epochs']}], Step [{i}/{len(dataloader)}], "
                        f"Loss: {loss.item():.4f}, Seg Loss: {seg_loss.item():.4f}, Det Loss: {det_loss.item():.4f}")
            
            avg_loss = running_loss / len(dataloader)
            avg_seg_loss = running_seg_loss / len(dataloader)
            avg_det_loss = running_det_loss / len(dataloader)
            print(f"\nEpoch [{epoch+1}/{self.training_config['epochs']}] Average Loss: {avg_loss:.4f} "
                f"(Seg: {avg_seg_loss:.4f}, Det: {avg_det_loss:.4f})")
    

            seg_val_loss, det_val_loss, val_loss = self.validate(model, val_dataloader, seg_criterion, yolo_loss_fn)
            print(f'Validation Loss: {val_loss.item():.4f} (Seg: {seg_val_loss.item():.4f}, Det: {det_val_loss.item():.4f})\n')

            if val_loss < min_loss:
                min_loss = val_loss
                # torch.save(model.state_dict(), 'full_model_weights.pth')
                # print('Model weights saved to full_model_weights.pth')
                improvement = 0
            else:
                improvement += 1

            if improvement >= self.training_config['early_stopping']:
                print('Early Stopping Activated')
                break

        print('Training complete')
        torch.save(model.state_dict(), 'full_model_weights_fin.pth')

if __name__ == '__main__':

    trainer = Trainer()

    trainer.train_full_model()
