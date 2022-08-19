# MapAI Compeition Template

This is a general template for the MapAI Competition and provides the 
users with a basic project using a FCN network backed by a pretrained backbone. 
There is also a simple submission python file displaying the structure of the 
required submission format and a suggested way of doing it.

The template is set up with config file for the Massachusetts Building Segmentation dataset, 
which can be found on the Kaggle website.
> https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset

The website requires to register a user in order to download the dataset.

Currently, the repository only supports aerial image segmentation. This will be expanded
to enable the use of laser data in combination with aerial images.

## Note

The dataset used in this template resizes the images from 1500x1500 down to 512x512. 
Resizing an image to a smaller format should be avoided as it reduces the amount of 
information available in the image - thus it might reduce the performance of the model.
A better approach would be to divide the image into tiles of smaller size, e.g. 9 tiles à 500x500.

The template does not implement data augmentation, which most likely would increase the performance
of the model. This is especially true if the dataset is small.

## Coming Soon

- Support for laser data + aerial image segmentation.