# Create dummy dataloader here for template competition?

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import cv2 as cv
import os

from yaml import load, dump, Loader, Dumper

from datautils import load_image, load_label, load_lidar

from datasets import load_dataset

class ImageAndLabelDataset(Dataset):

    def __init__(self,
                 opts: dict,
                 datatype: str = "validation"):

        self.opts = opts

        if "task" in datatype:
            auth_token = os.environ["ACCESS_TOKEN"]
            self.paths = load_dataset("sjyhne/mapai_evaluation_data", split=datatype, use_auth_token=auth_token)
        else:
            self.paths = load_dataset("sjyhne/mapai_training_data", split=datatype)

        print()

        print(f"Using number of images in {datatype}dataset: {int(self.paths.num_rows * self.opts['data_percentage'])}/{self.paths.num_rows}")

    def __len__(self):
        return int(self.paths.num_rows * self.opts["data_percentage"])

    def __getitem__(self, idx):
        pathdict = self.paths[idx]

        imagefilepath = pathdict["image"]
        labelfilepath = pathdict["mask"]

        assert imagefilepath.split("/")[-1] == labelfilepath.split("/")[-1], f"imagefilename and labelfilename does not match; {imagefilepath.split('/')[-1]} != {labelfilepath.split('/')[-1]}"

        filename = imagefilepath.split("/")[-1]

        image = load_image(imagefilepath, (self.opts["imagesize"], self.opts["imagesize"]))
        label = load_label(labelfilepath, (self.opts["imagesize"], self.opts["imagesize"]))

        assert image.shape[1:] == label.shape[:2], f"image and label shape not the same; {image.shape[1:]} != {label.shape[:2]}"

        return image, label, filename

class ImageLabelAndLidarDataset(Dataset):

    def __init__(self,
                 opts: dict,
                 datatype: str = "validation"):

        self.opts = opts

        if "task" in datatype:
            auth_token = os.environ["ACCESS_TOKEN"]
            self.paths = load_dataset("sjyhne/mapai_evaluation_data", split=datatype, use_auth_token=auth_token)
        else:
            self.paths = load_dataset("sjyhne/mapai_training_data", split=datatype)

        print(
            f"Using number of images in {datatype}dataset: {int(self.paths.num_rows * self.opts['data_percentage'])}/{self.paths.num_rows}")

    def __len__(self):
        return int(self.paths.num_rows * self.opts["data_percentage"])

    def __getitem__(self, idx):

        pathdict = self.paths[idx]

        imagefilepath = pathdict["image"]
        labelfilepath = pathdict["mask"]
        lidarfilepath = pathdict["lidar"]

        assert imagefilepath.split("/")[-1] == labelfilepath.split("/")[-1], f"imagefilename and labelfilename does not match; {imagefilepath.split('/')[-1]} != {labelfilepath.split('/')[-1]}"
        assert imagefilepath.split("/")[-1] == lidarfilepath.split("/")[-1], f"imagefilename and labelfilename does not match; {imagefilepath.split('/')[-1]} != {labelfilepath.split('/')[-1]}"

        filename = imagefilepath.split("/")[-1]

        image = load_image(imagefilepath, (self.opts["imagesize"], self.opts["imagesize"]))
        label = load_label(labelfilepath, (self.opts["imagesize"], self.opts["imagesize"]))
        lidar = load_lidar(lidarfilepath, (self.opts["imagesize"], self.opts["imagesize"]))

        assert image.shape[1:] == label.shape[:2], f"image and label shape not the same; {image.shape[1:]} != {label.shape[:2]}"
        assert image.shape[1:] == lidar.shape[:2], f"image and label shape not the same; {image.shape[1:]} != {label.shape[:2]}"

        # Concatenate lidar and image data
        lidar = lidar.unsqueeze(0)

        image = torch.cat((image, lidar), dim=0)

        return image, label, filename


class TestDataset(Dataset):
    def __init__(self,
                 opts: dict,
                 datatype: str = "test"):

        self.opts = opts

        self.imagepaths = get_paths_from_folder(opts[datatype]["imagefolder"])

        print(f"Number of images in {datatype}dataset: {len(self)}")

    def __len__(self):
        return len(self.imagepaths)

    def __getitem__(self, idx):
        imagefilepath = self.imagepaths[idx]

        filename = imagefilepath.split("/")[-1]

        image = load_image(imagefilepath, (self.opts["imagesize"], self.opts["imagesize"]))

        return image, filename


def create_dataloader(opts: dict, datatype: str = "test") -> DataLoader:

    if opts["task"] == 1:
        dataset = ImageAndLabelDataset(opts, datatype)
    elif opts["task"] == 2:
        dataset = ImageLabelAndLidarDataset(opts, datatype)

    dataloader = DataLoader(dataset, batch_size=opts[datatype]["batchsize"], shuffle=opts[datatype]["shuffle"])

    return dataloader

def create_testloader(opts: dict, datatype: str = "test") -> DataLoader:
    dataset = TestDataset(opts, datatype)

    testloader = DataLoader(dataset, batch_size=opts[datatype]["batchsize"], shuffle=opts[datatype]["shuffle"])

    return testloader

if __name__ == "__main__":

    opts = load(open("config/massachusetts.yaml"), Loader=Loader)

    testloader = create_dataloader(opts, "test")

    for batch in testloader:

        image, label, filename = batch

        print("image.shape:", image.shape)
        print("label.shape:", label.shape)
        print("filename:", filename)

        exit()