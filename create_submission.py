import argparse

import torch
import torchvision
import cv2 as cv

from yaml import load, Loader, dump, Dumper
from dataloader import create_testloader
from tqdm import tqdm

import os
import shutil

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--run", type=str, default="0", help="Which run number that should be used for inference")
    parser.add_argument("--config", type=str, default="config/massachusetts.yaml", help="Config")
    parser.add_argument("--task", type=int, default=1, help="Which task you are submitting for")
    parser.add_argument("--device", type=str, default="cpu", help="Which device the inference should run on")

    args = parser.parse_args()
    args.weights = os.path.join("runs", f"run_{args.run}", "best.pt")

    opts = load(open(args.config, "r"), Loader=Loader)
    try:
        opts = opts | vars(args)
    except Exception as e:
        opts = {**opts, **vars(args)}
    
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, aux_loss=True)
    
    model.load_state_dict(torch.load(opts["weights"]))

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

    testloader = create_testloader(opts, "test")

    predictionfolder = os.path.join(taskfolder, "predictions")

    os.mkdir(predictionfolder)

    device = opts["device"]

    model = model.to(device)

    for idx, batch in tqdm(enumerate(testloader), total=len(testloader), desc="Inference"):
        image, filename = batch
        filename = filename[0]
        image = image.to(device)

        prediction = model(image)["out"]
        prediction = torch.argmax(torch.softmax(prediction, dim=1), dim=1).squeeze().detach().numpy()

        filepath = os.path.join(predictionfolder, filename)

        for idx, value in enumerate(opts["classes"]):
            prediction[prediction == idx] = opts["class_to_color"][value]

        cv.imwrite(filepath, prediction)


    dump(opts, open(os.path.join(taskfolder, "opts.yaml"), "w"), Dumper=Dumper)




