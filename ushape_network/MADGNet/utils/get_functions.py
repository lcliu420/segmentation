import os
from datetime import datetime

import torch


def get_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('You are using "{}" device.'.format(device))
    return device


def get_deivce():
    return get_device()


def configure_dataset_args(args):
    args.dataset_type = args.dataset_name
    args.train_dataset_dir = os.path.join(args.data_path, args.dataset_type)
    args.val_dataset_dir = os.path.join(args.data_path, args.dataset_type)
    args.test_dataset_dir = os.path.join(args.data_path, args.dataset_type)
    if not args.image_dir_name:
        if args.dataset_name.lower().startswith("isic2018"):
            args.image_dir_name = "ISIC2018_Task1-2_Training_Input"
        else:
            args.image_dir_name = "images"
    if not args.mask_dir_name:
        if args.dataset_name.lower().startswith("isic2018"):
            args.mask_dir_name = "ISIC2018_Task1_Training_GroundTruth"
        else:
            args.mask_dir_name = "masks"
    if getattr(args, "save_path", None):
        args.output_root = args.save_path
    args.num_channels = 3
    args.num_classes = 1
    args.metric_list = ["DSC", "mIoU"]
    return args


def configure_isic_args(args):
    return configure_dataset_args(args)


def get_experiment_name(args):
    if getattr(args, "_resolved_experiment_name", None):
        return args._resolved_experiment_name

    if getattr(args, "experiment_name", None):
        args._resolved_experiment_name = args.experiment_name
        return args._resolved_experiment_name

    precision_tag = "amp" if torch.cuda.is_available() else "fp32"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args._resolved_experiment_name = "{}_madgnet_e{}_bs{}_{}x{}_cosine_{}_{}".format(
        args.dataset_name.lower(),
        args.epochs,
        args.batch_size,
        args.image_size,
        args.image_size,
        precision_tag,
        timestamp,
    )
    return args._resolved_experiment_name


def get_save_path(args):
    experiment_name = get_experiment_name(args)
    experiment_dir = os.path.join(args.output_root, experiment_name)
    for subdir in ["checkpoints", "reports", "predictions"]:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    args.experiment_dir = experiment_dir
    return experiment_dir
