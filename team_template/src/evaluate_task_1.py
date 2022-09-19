import argparse

import torch
import torchvision
import numpy as np
import cv2 as cv

from yaml import load, Loader, dump, Dumper
from dataloader import create_dataloader
from tqdm import tqdm
from eval_functions import iou, biou
import matplotlib.pyplot as plt

import gdown

import os
import shutil

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="config/data.yaml", help="Config")
    parser.add_argument("--task", type=int, default=1, help="Which task you are submitting for")
    parser.add_argument("--device", type=str, default="cpu", help="Which device the inference should run on")
    parser.add_argument("--weights", type=str, help="Path to weights for the specific model and task")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--data_percentage", type=float, default=1.0, help="Percentage of the whole dataset that is used")
    parser.add_argument("--dtype", type=str, default="validation", help="Which data to test against")

    args = parser.parse_args()

    opts = load(open(args.config, "r"), Loader=Loader)
    try:
        opts = opts | vars(args)
    except Exception as e:
        opts = {**opts, **vars(args)}

    model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=opts["num_classes"])

    # Download trained model ready for inference
    pt_share_link = "https://drive.google.com/file/d/17YB5-KZVW-mqaQdz4xv7rioDr4DzhfOU/view?usp=sharing"
    pt_id = pt_share_link.split("/")[-2]

    # Download trained model ready for inference
    url_to_drive = f"https://drive.google.com/uc?id={pt_id}"
    output_file = "pretrained_task1.pt"

    gdown.download(url_to_drive, output_file, quiet=False)

    # Commented for testing purposes
    model.load_state_dict(torch.load(output_file))

    submissionfolder = "submission"

    taskfolder = os.path.join(submissionfolder, "task" + str(opts["task"]))

    if not os.path.exists(submissionfolder):
        os.mkdir(submissionfolder)

    if os.path.exists(taskfolder):
        answer = input(f"Are you sure you want to delete folder: '{taskfolder}'? (y/n) ")
        if answer == "y":
            shutil.rmtree(taskfolder)
        else:
            print("Exiting because you do not want to delete folder:", taskfolder)
            exit(0)

    os.mkdir(taskfolder)

    testloader = create_dataloader(opts, opts["dtype"])

    predictionfolder = os.path.join(taskfolder, "predictions")

    os.mkdir(predictionfolder)

    device = opts["device"]

    model = model.to(device)

    iou_scores = np.ndarray((len(testloader)))
    biou_scores = np.ndarray((len(testloader)))

    model.eval()

    for idx, batch in tqdm(enumerate(testloader), total=len(testloader), desc="Inference", leave=False):
        image, label, filename = batch
        image = image.to(device)
        label = label.to(device)

        filename = filename[0]

        prediction = model(image)["out"]
        prediction = torch.argmax(torch.softmax(prediction, dim=1), dim=1).squeeze().detach().numpy()

        label = label.squeeze().detach().numpy()

        prediction = np.uint8(prediction)
        label = np.uint8(label)

        assert prediction.shape == label.shape, f"Prediction and label shape is not same, pls fix [{prediction.shape} - {label.shape}]"

        iou_score = iou(prediction, label)
        biou_score = biou(label, prediction)

        iou_scores[idx] = np.round(iou_score, 6)
        biou_scores[idx] = np.round(biou_score, 6)

        filepath = os.path.join(predictionfolder, filename)

        prediction_visual = np.copy(prediction)

        for idx, value in enumerate(opts["classes"]):
            prediction_visual[prediction_visual == idx] = opts["class_to_color"][value]

        image = image.squeeze().detach().numpy()[:3, :, :].transpose(1, 2, 0)

        fig, ax = plt.subplots(1, 3)
        columns = 3
        rows = 1

        ax[0].set_title("Input (RGB)")
        ax[0].imshow(image)
        ax[1].set_title("Prediction")
        ax[1].imshow(prediction_visual)
        ax[2].set_title("Label")
        ax[2].imshow(label)

        plt.savefig(filepath.split(".")[0] + ".png")
        cv.imwrite(filepath.split(".")[0] + ".png", prediction_visual)

    print("iou_score:", np.round(iou_scores.mean(), 5), "biou_score:", np.round(biou_scores.mean(), 5))

    dump(opts, open(os.path.join(taskfolder, "opts.yaml"), "w"), Dumper=Dumper)




