import cv2 as cv
import numpy as np

import argparse
import os
import json
from pathlib import Path

from eval_functions import iou, biou

def calculate_score(preds: np.array, tars: np.array) -> dict:

    assert preds.shape == tars.shape, f"pred shape {preds.shape} does not match tar shape {tars.shape}"
    assert len(preds.shape) != 4, f"expected shape is (bs, ydim, xdim), but found {preds.shape}"
    assert type(preds) == np.ndarray, f"preds is a {type(preds)}, but should be numpy.ndarray"
    assert type(tars) == np.ndarray, f"tars is a {type(tars)}, but should be numpy.ndarray"
    assert type(preds[0][0][0]) == np.uint8, f"preds is not of type np.uint8, but {type(preds[0][0][0])}"
    assert type(tars[0][0][0]) == np.uint8, f"tars is not of type np.uint8, but {type(tars[0][0][0])}"

    bs = preds.shape[0]

    t_iou = 0
    t_biou = 0

    for i in range(bs):
        t_iou += iou(preds[i], tars[i])
        t_biou += biou(preds[i], tars[i])

    t_iou /= bs
    t_biou /= bs

    score = (t_iou + t_biou) / 2

    return {"score": score, "iou": t_iou, "biou": t_biou}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=int, default=1, help="The task to evaluate")
    parser.add_argument("--data_percentage", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="validation", help="Which data to test against")

    args = parser.parse_args()

    sf = os.path.join("submission", "task" + str(args.task), "predictions")
    if args.dtype != "validation":
        lf = os.path.join(f"../../data/task{args.task}_test/", "masks")
    else:
        lf = os.path.join("../../data/", args.dtype, "masks")

    pred_files = os.listdir(sf)
    mask_files = os.listdir(lf)

    pred_paths = [os.path.join(sf, x) for x in os.listdir(sf) if x.split(".")[-1] == "tif"]
    mask_paths = [os.path.join(lf, x.split("/")[-1]) for x in pred_paths]

    # Check that num preds is equal num masks
    assert len(pred_paths) == len(mask_paths), f"There are not an equal amount of masks and preds {len(mask_paths)}, {len(pred_paths)}"

    pred_paths = sorted(pred_paths)
    mask_paths = sorted(mask_paths)

    iou_scores = np.ndarray((len(pred_paths),))
    biou_scores = np.ndarray((len(pred_paths),))

    for i in range(len(pred_paths)):

        pred_path = pred_paths[i]
        mask_path = mask_paths[i]

        pred_file = pred_path.split("/")[-1]
        mask_file = mask_path.split("/")[-1]

        assert pred_file == mask_file, f"preds names does not match mask names {pred_file}, {mask_file}"

        pred = cv.imread(pred_path, cv.IMREAD_GRAYSCALE)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

        iouscore = iou(pred, mask)
        biouscore = biou(mask, pred)

        iou_scores[i] = iouscore
        biou_scores[i] = biouscore

    iscore = np.round(iou_scores.sum()/len(iou_scores), 4)
    bscore = np.round(biou_scores.sum()/len(biou_scores), 4)

    if np.isnan(iscore):
        print("iou is nan")
        iscore = 0.0

    if np.isnan(bscore):
        print("biou is nan")
        bscore = 0.0

    print(f"Evaluation {str(Path('../..').parent.absolute())} Task {args.task} -", "IoU:", iscore, "BIoU:", bscore)

    result_file = f"results_task_{args.task}.json"

    result_dict = {"iou": iscore, "biou": bscore}

    json.dump(result_dict, open(result_file, "w"))

