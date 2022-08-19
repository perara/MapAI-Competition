import os

import numpy as np
from yaml import load, dump, Loader, Dumper
from tqdm import tqdm
import torch
import torchvision
from tabulate import tabulate

import argparse
import time
import math

from dataloader import create_dataloader
from utils import create_run_dir, store_model_weights, record_scores
from eval_functions import calculate_score

def test(opts, dataloader, model, lossfn):

    model.eval()

    device = opts["device"]

    losstotal = np.zeros((len(dataloader)), dtype=float)
    ioutotal = np.zeros((len(dataloader)), dtype=float)
    bioutotal = np.zeros((len(dataloader)), dtype=float)
    scoretotal = np.zeros((len(dataloader)), dtype=float)


    for idx, batch in tqdm(enumerate(dataloader), leave=False, total=len(dataloader), desc="Test"):
        image, label, filename = batch
        image = image.to(device)
        label = label.to(device)

        output = model(image)["out"]

        loss = lossfn(output, label).item()

        output = torch.argmax(torch.softmax(output, dim=1), dim=1)
        metrics = calculate_score(output.detach().numpy().astype(np.uint8), label.detach().numpy().astype(np.uint8))

        losstotal[idx] = loss
        ioutotal[idx] = metrics["iou"]
        bioutotal[idx] = metrics["biou"]
        scoretotal[idx] = metrics["score"]

    loss = round(losstotal.mean(), 4)
    iou = round(ioutotal.mean(), 4)
    biou = round(bioutotal.mean(), 4)
    score = round(scoretotal.mean(), 4)

    return loss, iou, biou, score

def train(opts):

    device = opts["device"]

    model = torchvision.models.segmentation.fcn_resnet50(num_classes=opts["num_classes"], backbone_weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model.to(device)
    model = model.float()

    optimizer = torch.optim.Adam(model.parameters(), lr=opts["lr"])
    lossfn = torch.nn.CrossEntropyLoss()

    epochs = opts["epochs"]

    trainloader = create_dataloader(opts, "test")
    valloader = create_dataloader(opts, "val")

    bestscore = math.inf

    for e in range(epochs):

        model.train()

        losstotal = np.zeros((len(trainloader)), dtype=float)
        scoretotal = np.zeros((len(trainloader)), dtype=float)
        ioutotal = np.zeros((len(trainloader)), dtype=float)
        bioutotal = np.zeros((len(trainloader)), dtype=float)

        stime = time.time()

        for idx, batch in tqdm(enumerate(trainloader), leave=True, total=len(trainloader), desc="Train", position=0):
            image, label, filename = batch
            image = image.to(device)
            label = label.to(device)

            output = model(image)["out"]

            loss = lossfn(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossitem = loss.item()
            output = torch.argmax(torch.softmax(output, dim=1), dim=1)
            trainmetrics = calculate_score(output.detach().numpy().astype(np.uint8), label.detach().numpy().astype(np.uint8))

            losstotal[idx] = lossitem
            ioutotal[idx] = trainmetrics["iou"]
            bioutotal[idx] = trainmetrics["biou"]
            scoretotal[idx] = trainmetrics["score"]

        testloss, testiou, testbiou, testscore = test(opts, valloader, model, lossfn)
        trainloss = round(losstotal.mean(), 4)
        trainiou = round(ioutotal.mean(), 4)
        trainbiou = round(bioutotal.mean(), 4)
        trainscore = round(scoretotal.mean(), 4)

        if testscore < bestscore:
            bestscore = testscore
            store_model_weights(opts, model, "best")
        else:
            store_model_weights(opts, model, "last")


        print("")
        print(tabulate([["train", trainloss, trainiou, trainbiou, trainscore], ["test", testloss, testiou, testbiou, testscore]], headers=["Type", "Loss", "IoU", "BIoU", "Score"]))

        scoredict = {
            "epoch": e,
            "trainloss": trainloss,
            "testloss": testloss,
            "trainiou": trainiou,
            "testiou": testiou,
            "trainbiou": trainbiou,
            "testbiou": testbiou,
            "trainscore": trainscore,
            "testscore": testscore
        }

        record_scores(opts, scoredict)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Training a segmentation model")

    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate used during training")
    parser.add_argument("--config", type=str, default="config/massachusetts.yaml", help="Configuration file to be used")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    # Import config
    opts = load(open(args.config, "r"), Loader)

    # Combine args and opts in single dict
    opts = opts | vars(args)

    print("Opts:", opts)

    rundir = create_run_dir(opts)
    opts["rundir"] = rundir
    dump(opts, open(os.path.join(rundir, "opts.yaml"), "w"), Dumper)

    train(opts)

