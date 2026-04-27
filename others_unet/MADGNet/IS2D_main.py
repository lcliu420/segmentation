#-*- coding:utf-8 -*-

import argparse
import warnings

from IS2D_Experiment.biomedical_2dimage_segmentation_experiment import ISICSegmentationExperiment
from utils.get_functions import configure_dataset_args, get_save_path
from utils.save_functions import save_metrics

warnings.filterwarnings("ignore")


def build_parser():
    parser = argparse.ArgumentParser(description="Train and evaluate MADGNet on a medical image segmentation dataset.")
    parser.add_argument("--data_path", type=str, default="dataset/BioMedicalDataset")
    parser.add_argument("--dataset_name", type=str, default="ISIC2018")
    parser.add_argument("--image_dir_name", type=str, default=None)
    parser.add_argument("--mask_dir_name", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--image_size", type=int, default=352)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--seed_fix", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--eval_split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--eval_epoch", type=int, default=None)
    parser.add_argument("--save_predictions", default=False, action="store_true")
    parser.add_argument("--backbone_pretrained", default=True, action=argparse.BooleanOptionalAction)

    parser.add_argument("--cnn_backbone", type=str, default="resnest50")
    parser.add_argument("--scale_branches", type=int, default=3)
    parser.add_argument("--frequency_branches", type=int, default=16)
    parser.add_argument("--frequency_selection", type=str, default="top")
    parser.add_argument("--block_repetition", type=int, default=1)
    parser.add_argument("--min_channel", type=int, default=32)
    parser.add_argument("--min_resolution", type=int, default=8)

    return parser


def main():
    args = build_parser().parse_args()
    args = configure_dataset_args(args)
    experiment_dir = get_save_path(args)

    print("Hello! We start the MADGNet experiment on {}.".format(args.dataset_name))
    print("Experiment outputs will be saved to {}.".format(experiment_dir))
    experiment = ISICSegmentationExperiment(args)

    if args.train:
        experiment.train()

    test_results, checkpoint_name = experiment.inference(split=args.eval_split)
    model_dirs = get_save_path(args)

    print("Save MADGNet Test Results...")
    save_metrics(args, test_results, model_dirs, split=args.eval_split, checkpoint_name=checkpoint_name)


if __name__ == "__main__":
    main()
