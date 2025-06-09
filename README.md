# YoloV1 UNet Semantic Segmentation Project

## Running Instructions:

### Installing Requirements
Create a new virtual environment and run the following command:

```bash
pip install -r requirements.txt
```

### Generate annotations:
The annotations used in obtaining the results have already been included in this project, but if you want to generate them again, please run the dataloader.py python file, ensuring that the CamVid_manager() function has been uncommented and the two directory variables have been changed as shown above

### Training the model:
Modify any desired hyperparameters in the training_params section of config.yaml and then simply run the train.py file. The model weights will then be saved to your current working directory:

- root_dir: [str] directory of the data folder
- epochs: [int] number of training epochs
- batch_size: [int] number of batches used in training
- transforms: [bool] if augmentation should be applied to the training dataset
- learning_rate: [float] optimiser lr
- segmentation_loss_weight: [float] by how much the segmentation loss functions are weighted (1 for equal weighting, 0 to only train the detection model)
- detection_loss_weight: [float] by how much the detection loss functions are weighted (1 for equal weighting, 0 to only train the segmentation model)
- shuffle: [bool] shuffles the batches
- num_workers: [int] dataloader number of workers
- early_stopping: [int] stops after number of epochs with no improvement by validation set
- low_instance_cls: [list of ints] classes with low instances in dataset. Set as empty list for no class preference
	1: car
	2: pedestrian
	3: bicyclist
	4: motorcycle/scooter
	5: truck/ bus
- flip_prob: [float] probability that classes indicated in low_instance_cls will be augmented by random horizontal flip
- rot_prob: [float] probability that classes indicated in low_instance_cls will be augmented by random rotation
- base_flip_prob: [float] base probability that image flips around the y axis (applied to ALL images, set to 0 for no augmentation)
- base_rot_prob: [float] base probability that image gets rotated by a maximum of 15 degrees (applied to ALL images, set to 0 for no augmentation)

# Testing the model:
With the resultant model weights in the working directory (or modifying line 467 to specify other weight path), run test.py for model metrics, as well as random sample images to be generated and saved to the working directory. Modifying line 478 allows the test/ train/ val split to be specified



