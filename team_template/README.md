# MapAI Compeition Template

This is a general template for the MapAI Competition and provides the 
users with a basic project using a FCN network backed by a pretrained backbone. 
There is also a simple submission python file displaying the structure of the 
required submission format and a suggested way of doing it.

Currently, the repository only supports aerial image segmentation. This will be expanded
to enable the use of laser data in combination with aerial images.

## Data Folder Structure

```bash
data
|--images
   |--train
      |--xyz.tif
   |--test
      |--xyz.tif
   |--val
      |--xyz.tif
|--lidar
   |--train
      |--xyz.tif
   |--test
      |--xyz.tif
   |--val
      |--xyz.tif
|--points
   |--train
      |--xyz.laz
   |--test
      |--xyz.laz
   |--val
      |--xyz.laz
|--masks
   |--train
      |--xyz.tif
   |--val
      |--xyz.tif

```

## Note

The dataset used in this template resizes the images from 1500x1500 down to 512x512. 
Resizing an image to a smaller format should be avoided as it reduces the amount of 
information available in the image - thus it might reduce the performance of the model.
A better approach would be to divide the image into tiles of smaller size, e.g. 9 tiles à 500x500.

The template does not implement data augmentation, which most likely would increase the performance
of the model. This is especially true if the dataset is small.

The template comes with a preset configuration for the train, val, and test data splits. You are
perfectly allowed to change the train and val split as you like, but the test split must be
unaltered.

The point folder contains raw output from the lidar in a .laz file. You can read more about the
file format and discover a python package for reading and writing at
> https://pylas.readthedocs.io/en/latest/

## Coming Soon

- Support for laser data + aerial image segmentation.

## Team info (Fill in the blank fields):

Team name: ___

Team participants:  ___

Emails: ___

Countr(y/ies): ___