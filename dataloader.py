# Create dummy dataloader here for template competition?

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import cv2 as cv

from yaml import load, dump, Loader, Dumper

from datautils import get_paths_from_folder, load_image, load_label

class ImageAndLabelDataset(Dataset):

    def __init__(self,
                 opts: dict,
                 datatype: str = "val"):

        self.opts = opts

        self.imagepaths = get_paths_from_folder(opts[datatype]["imagefolder"])
        self.labelpaths = get_paths_from_folder(opts[datatype]["labelfolder"])

        assert len(self.imagepaths) == len(self.labelpaths), f"Length of imagepaths does not match length of labelpaths; {len(self.imagepaths)} != {len(self.labelpaths)}"

        print(f"Number of images in {datatype}dataset: {len(self)}")

    def __len__(self):
        return len(self.imagepaths)

    def __getitem__(self, idx):
        imagefilepath = self.imagepaths[idx]
        labelfilepath = self.labelpaths[idx]

        assert imagefilepath.split("/")[-1] == labelfilepath.split("/")[-1], f"imagefilename and labelfilename does not match; {imagefilepath.split('/')[-1]} != {labelfilepath.split('/')[-1]}"

        filename = imagefilepath.split("/")[-1]

        image = load_image(imagefilepath, (self.opts["imagesize"], self.opts["imagesize"]))
        label = load_label(labelfilepath, (self.opts["imagesize"], self.opts["imagesize"]))

        assert image.shape[1:] == label.shape[:2], f"image and label shape not the same; {image.shape[1:]} != {label.shape[:2]}"

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
    dataset = ImageAndLabelDataset(opts, datatype)

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