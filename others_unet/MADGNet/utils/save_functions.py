import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

from utils.get_functions import get_save_path


def save_checkpoint(args, model, optimizer, scheduler, epoch, best_score, is_best=False, is_last=True, save_epoch_snapshot=False):
    experiment_dir = get_save_path(args)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")

    checkpoint = {
        "epoch": epoch,
        "best_score": best_score,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    if is_last:
        torch.save(checkpoint, os.path.join(checkpoint_dir, "model_last.pth.tar"))

    if is_best:
        torch.save(checkpoint, os.path.join(checkpoint_dir, "model_best.pth.tar"))

    if save_epoch_snapshot:
        torch.save(checkpoint, os.path.join(checkpoint_dir, "model_weight(EPOCH {}).pth.tar".format(epoch)))


def save_history(args, history_frame):
    experiment_dir = get_save_path(args)
    save_path = os.path.join(experiment_dir, "history.csv")
    history_frame.to_csv(save_path, index=False)
    print("history csv file is saved at {}".format(save_path))


def save_metrics(args, test_results, experiment_dir, split, checkpoint_name):
    print("###################### TEST REPORT ######################")
    print("Split         :	 {}".format(split))
    print("Checkpoint    :	 {}".format(checkpoint_name))
    for metric in test_results.keys():
        print("Mean {}    :	 {}".format(metric, test_results[metric]))
    print("###################### TEST REPORT ######################\n")

    report_dir = os.path.join(experiment_dir, "reports")
    test_results_save_path = os.path.join(report_dir, "test_report({}-{}).txt".format(split, checkpoint_name.replace(".pth.tar", "")))

    with open(test_results_save_path, "w") as handle:
        handle.write("###################### TEST REPORT ######################\n")
        handle.write("Split         :	 {}\n".format(split))
        handle.write("Checkpoint    :	 {}\n".format(checkpoint_name))
        for metric in test_results.keys():
            handle.write("Mean {}    :	 {}\n".format(metric, test_results[metric]))
        handle.write("###################### TEST REPORT ######################\n")

    print("test results txt file is saved at {}".format(test_results_save_path))


def save_prediction_masks(args, prediction_items, split):
    experiment_dir = get_save_path(args)
    prediction_dir = os.path.join(experiment_dir, "predictions", split)
    os.makedirs(prediction_dir, exist_ok=True)

    for name, prediction in prediction_items:
        mask = (prediction.squeeze().numpy() >= 0.5).astype(np.uint8) * 255
        Image.fromarray(mask).save(os.path.join(prediction_dir, "{}.png".format(name)))

    print("prediction masks are saved at {}".format(prediction_dir))
