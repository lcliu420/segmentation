import os

import torch

from utils.get_functions import get_save_path


def resolve_checkpoint_path(args):
    if args.checkpoint_path is not None:
        return args.checkpoint_path

    if not getattr(args, "experiment_name", None) and not getattr(args, "_resolved_experiment_name", None):
        raise FileNotFoundError("No checkpoint found. Provide --checkpoint_path or --experiment_name.")

    experiment_dir = get_save_path(args)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")

    if args.eval_epoch is not None:
        return os.path.join(checkpoint_dir, "model_weight(EPOCH {}).pth.tar".format(args.eval_epoch))

    best_path = os.path.join(checkpoint_dir, "model_best.pth.tar")
    last_path = os.path.join(checkpoint_dir, "model_last.pth.tar")

    if os.path.exists(best_path):
        return best_path
    if os.path.exists(last_path):
        return last_path

    raise FileNotFoundError("No checkpoint found. Train the model first or provide --checkpoint_path.")


def load_model(args, model, optimizer=None, scheduler=None):
    load_path = resolve_checkpoint_path(args)

    print("Your model is loaded from {}.".format(load_path))
    checkpoint = torch.load(load_path, map_location=args.device)
    print(".pth.tar keys() = {}.".format(checkpoint.keys()))

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    checkpoint["checkpoint_path"] = load_path
    return checkpoint
